// Load custom ADM2 shapefile for Timor-Leste
var timor_leste_adm2 = ee.FeatureCollection('projects/ee-spenson/assets/tls_admn_ad2_py_s2_unocha_pp');

// Center map on Timor-Leste
Map.centerObject(timor_leste_adm2, 8);

// Get unique ADM1 codes
var adm1_codes = timor_leste_adm2.aggregate_array('ADM1_PCODE').distinct();

// Function to get cropland mask
function getCroplandMask(adm2) {
  var cropland = ee.Image('USGS/GFSAD1000_V1').select('landcover').clip(adm2);
  return cropland.eq(1).or(cropland.eq(2)).or(cropland.eq(3))
                 .or(cropland.eq(4)).or(cropland.eq(5));
}

// Cloud and shadow masking for Landsat
function maskCloudsAndShadows(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 4).eq(0);
  var shadow = qa.bitwiseAnd(1 << 3).eq(0);
  return image.updateMask(cloud).updateMask(shadow);
}

// NDVI calculation
function calculateNDVI(image, cropland_mask) {
  var nir = image.select('SR_B5').multiply(0.0000275).add(-0.2);
  var red = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  return image.addBands(ndvi).updateMask(cropland_mask);
}

// Monthly NDVI metric computation
function calculateMonthlyMetrics(adm1_codes, year, month) {
  var start_date = ee.Date.fromYMD(year, month, 1);
  var end_date = start_date.advance(1, 'month');
  var metrics_fc = ee.FeatureCollection([]);

  adm1_codes.forEach(function(adm1_code) {
    var adm2_filtered = timor_leste_adm2.filter(ee.Filter.eq('ADM1_PCODE', adm1_code));
    var cropland_mask = getCroplandMask(adm2_filtered);

    var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
      .filterDate(start_date, end_date)
      .filterBounds(adm2_filtered)
      .filter(ee.Filter.lt('CLOUD_COVER', 90))
      .map(maskCloudsAndShadows)
      .map(function(image) { return calculateNDVI(image, cropland_mask); });

    var mean_ndvi = collection.mean().select('NDVI').clip(adm2_filtered).rename('mean_ndvi');
    var median_ndvi = collection.median().select('NDVI').clip(adm2_filtered).rename('median_ndvi');

    var abs_diff = collection.map(function(image) {
      return image.select('NDVI').subtract(median_ndvi).abs().rename('abs_diff');
    });

    var mad_ndvi = abs_diff.median().select('abs_diff').rename('mad_ndvi');
    var robust_std_ndvi = mad_ndvi.multiply(1.4826).rename('robust_std_ndvi');

    var z_score_median_ndvi = collection.map(function(image) {
      return image.expression(
        '(NDVI - median) / robust_std',
        {
          'NDVI': image.select('NDVI'),
          'median': median_ndvi,
          'robust_std': robust_std_ndvi
        }
      ).rename('z_score_median_ndvi');
    }).mean().clip(adm2_filtered);

    var metrics = mean_ndvi.addBands([
      median_ndvi, mad_ndvi, robust_std_ndvi, z_score_median_ndvi
    ]);

    var metrics_per_region = metrics.reduceRegions({
      collection: adm2_filtered,
      reducer: ee.Reducer.mean(),
      scale: 30
    }).map(function(feature) {
      return feature.set({
        'year': year,
        'month': month,
        'mean_ndvi': feature.get('mean_ndvi'),
        'median_ndvi': feature.get('median_ndvi'),
        'mad_ndvi': feature.get('mad_ndvi'),
        'robust_std_ndvi': feature.get('robust_std_ndvi'),
        'z_score_median_ndvi': feature.get('z_score_median_ndvi')
      });
    });

    metrics_fc = metrics_fc.merge(metrics_per_region);
  });

  return metrics_fc;
}

// Export NDVI metrics to Drive
function exportMonthlyMetricsAsSingleCSV(startYear, startMonth, endYear, endMonth) {
  var all_metrics_fc = ee.FeatureCollection([]);

  adm1_codes.evaluate(function(codes) {
    var currentYear = startYear;
    var currentMonth = startMonth;

    while (currentYear < endYear || (currentYear === endYear && currentMonth <= endMonth)) {
      var monthly_metrics = calculateMonthlyMetrics(codes, currentYear, currentMonth);
      all_metrics_fc = all_metrics_fc.merge(monthly_metrics);

      if (currentMonth === 12) {
        currentMonth = 1;
        currentYear += 1;
      } else {
        currentMonth += 1;
      }
    }

    Export.table.toDrive({
      collection: all_metrics_fc,
      description: 'timor_leste_ndvi_metrics_combined',
      folder: 'GEE_timor_leste_Export_Landsat',
      fileFormat: 'CSV',
      selectors: ['year', 'month', 'ADM1_PCODE', 'ADM2_PCODE', 'mean_ndvi', 'median_ndvi', 'mad_ndvi', 'robust_std_ndvi', 'z_score_median_ndvi']
    });
  });
}

// Cropland ratio calculation
function calculateCroplandRatio(adm2) {
  var cropland_mask = getCroplandMask(adm2);

  var cropland_area = cropland_mask.multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: adm2.geometry(),
      scale: 30,
      maxPixels: 1e9
    }).get('landcover');

  var total_area = ee.Image.pixelArea()
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: adm2.geometry(),
      scale: 30,
      maxPixels: 1e9
    }).get('area');

  var cropland_ratio = ee.Number(cropland_area).divide(ee.Number(total_area)).multiply(100);
  return cropland_ratio;
}

// Apply to all ADM2s
function calculateCroplandRatiosForADM2() {
  return timor_leste_adm2.map(function(adm2) {
    var cropland_ratio = calculateCroplandRatio(adm2);
    return adm2.set('cropland_ratio', cropland_ratio);
  });
}

// Export cropland ratios and visualize
function exportCroplandRatios() {
  var cropland_ratios = calculateCroplandRatiosForADM2();

  // Convert cropland_ratio to image
  var cropland_ratio_image = cropland_ratios
    .reduceToImage({
      properties: ['cropland_ratio'],
      reducer: ee.Reducer.first()
    })
    .clip(timor_leste_adm2);

  var visParams = {
    min: 0,
    max: 100,
    palette: ['ffffff', 'a1dab4', '41b6c4', '2c7fb8', '253494']
  };

  // Add visualization to map
  Map.addLayer(cropland_ratio_image, visParams, 'Cropland Ratio (%)');

  // Overlay ADM2 borders
  Map.addLayer(
    timor_leste_adm2.style({color: '000000', fillColor: '00000000', width: 1}),
    {},
    'ADM2 Boundaries'
  );

  print('Cropland Ratios by ADM2', cropland_ratios);

  Export.table.toDrive({
    collection: cropland_ratios,
    description: 'timor_adm2_cropland_ratios',
    folder: 'GEE_timor_leste_Cropland_Ratios',
    fileFormat: 'CSV',
    selectors: ['ADM2_PCODE', 'ADM1_PCODE', 'cropland_ratio']
  });
}

// Set export date range
var startYear = 2018;
var startMonth = 11;
var endYear = 2024;
var endMonth = 10;

// Run exports
exportMonthlyMetricsAsSingleCSV(startYear, startMonth, endYear, endMonth);
exportCroplandRatios();
