// 0. Inputs & Settings
var adm2FC = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/LULC_Mean_Probability_Harvest_cropland_mask_adm2'
  //'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp'
);
var START_YEAR = 2019;
var END_YEAR   = 2025;

// 1. Load & cloud‐mask Sentinel-2 SR (keep bands B8, B4, B2 for NDVI & EVI)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .map(function(img) {
    var scl = img.select('SCL');
    // SCL < 7 = clear (masks clouds/shadows)
    return img.updateMask(scl.lt(7));
  })
  .select(['B8','B4','B2']);

// 2. Build server-side list of month-start dates
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var monthCount = ee.Date.fromYMD(END_YEAR, 12, 1)
  .difference(firstOfJan, 'month')
  .add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) {
    return firstOfJan.advance(n, 'month');
  });

// 3. Loop over each ADM2 region to create one export per ADM2_PCODE
var adm2Codes = adm2FC.aggregate_array('ADM2_PCODE').getInfo();
adm2Codes.forEach(function(adm2_code) {
  // Grab the single feature & its geometry
  var feat = adm2FC.filter(ee.Filter.eq('ADM2_PCODE', adm2_code)).first();
  var regionGeom = feat.geometry();

  // Build a FeatureCollection of monthly stats
  var metrics = ee.FeatureCollection(
    monthDates.map(function(d) {
      d = ee.Date(d);
      var start = d;
      var end   = d.advance(1, 'month').advance(-1, 'second');
      var dateStr = d.format('YYYY-MM-dd');

      // Filter and cloud-mask for this month & region
      var col = s2Sr
        .filterDate(start, end)
        .filterBounds(regionGeom);

      // Count how many images
      var count = col.size();

      // NDVI series
      var ndviCol = col.map(function(img) {
        return img.normalizedDifference(['B8','B4']).rename('NDVI');
      });

      // EVI series: 2.5*(NIR−RED)/(NIR+6*RED−7.5*BLUE+1)
      var eviCol = col.map(function(img) {
        return img.expression(
          '2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)',
          {
            NIR:  img.select('B8'),
            RED:  img.select('B4'),
            BLUE: img.select('B2')
          }
        ).rename('EVI');
      });

      // Compute composites
      var meanNdvi = ndviCol.mean();
      var maxNdvi  = ndviCol.max();
      var meanEvi  = eviCol.mean();
      var maxEvi   = eviCol.max();

      // Reduce to get mean & max values
      var stats = ee.Dictionary(
        meanNdvi
          .addBands(maxNdvi)
          .addBands(meanEvi)
          .addBands(maxEvi)
          .reduceRegion({
            reducer:   ee.Reducer.mean()
                          .combine({reducer2: ee.Reducer.max(), sharedInputs: true}),
            geometry:  regionGeom,
            scale:     10,
            maxPixels: 1e13,
            tileScale: 4
          })
      );

      // Safely extract each stat (null if missing)
      var mn = ee.Algorithms.If(stats.contains('NDVI_mean'),
                                stats.get('NDVI_mean'), null);
      var mx = ee.Algorithms.If(stats.contains('NDVI_max'),
                                stats.get('NDVI_max'), null);
      var me = ee.Algorithms.If(stats.contains('EVI_mean'),
                                stats.get('EVI_mean'), null);
      var xe = ee.Algorithms.If(stats.contains('EVI_max'),
                                stats.get('EVI_max'), null);

      // Build feature with only desired properties
      return ee.Feature(null, {
        ADM2_PCODE:   adm2_code,
        date:         dateStr,
        count_images: count,
        mean_NDVI:    mn,
        max_NDVI:     mx,
        mean_EVI:     me,
        max_EVI:      xe
      });
    })
  );

  // 4. Export to Drive with explicit selectors (drops geometry column)
  Export.table.toDrive({
    collection:     metrics,
    description:    'Metrics_' + adm2_code,
    fileNamePrefix: adm2_code + '_NDVI_EVI_monthly',
    folder:         'EarthEngineExports_NDVI_EVI',
    fileFormat:     'CSV',
    selectors: [
      'ADM2_PCODE',
      'date',
      'count_images',
      'mean_NDVI',
      'max_NDVI',
      'mean_EVI',
      'max_EVI'
    ]
  });
});
