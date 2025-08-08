// 0. Inputs & Settings
var adm2FC = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/LULC_Mean_Probability_Harvest_cropland_mask_adm2'
//  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp'
);

var START_YEAR = 2019;
var END_YEAR   = 2025;

// 1. Load & cloud‐mask Sentinel-2 SR once (keep only B8 & B4)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .map(function(img) {
    var clear = img.select('SCL').lt(7);
    return img.updateMask(clear);
  })
  .select(['B8', 'B4']);

// 2. Build list of monthly start dates
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var firstOfDec = ee.Date.fromYMD(END_YEAR, 12, 1);
var monthCount = firstOfDec.difference(firstOfJan, 'month').add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) { return firstOfJan.advance(n, 'month'); });

// 3. Process each ADM2 feature individually
adm2FC.aggregate_array('ADM2_PCODE').getInfo().forEach(function(adm2_code) {
  var feature = adm2FC.filter(ee.Filter.eq('ADM2_PCODE', adm2_code)).first();
  var regionGeom = ee.Feature(feature).geometry();

  // Compute NDVI metrics per month
  var metrics = ee.FeatureCollection(monthDates.map(function(d) {
    d = ee.Date(d);
    var year  = d.get('year');
    var month = d.get('month');
    var start = d;
    var end   = start.advance(1, 'month').advance(-1, 'second');

    var col = s2Sr.filterDate(start, end).filterBounds(regionGeom);

    var ndviCol = col.map(function(img) {
      return img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    });

    var meanImg = ndviCol.mean();
    var maxImg  = ndviCol.max();

    var meanDict = meanImg.reduceRegion({
      reducer:   ee.Reducer.mean(),
      geometry:  regionGeom,
      scale:     10,
      maxPixels: 1e13,
      tileScale: 4
    });

    var maxDict = maxImg.reduceRegion({
      reducer:   ee.Reducer.max(),
      geometry:  regionGeom,
      scale:     10,
      maxPixels: 1e13,
      tileScale: 4
    });

    var meanVal = ee.Algorithms.If(
      ee.Dictionary(meanDict).contains('NDVI'),
      ee.Dictionary(meanDict).get('NDVI'),
      null
    );
    var maxVal = ee.Algorithms.If(
      ee.Dictionary(maxDict).contains('NDVI'),
      ee.Dictionary(maxDict).get('NDVI'),
      null
    );

    return ee.Feature(null, {
      ADM2_PCODE: adm2_code,
      year:       year,
      month:      month,
      mean_NDVI:  meanVal,
      max_NDVI:   maxVal
    });
  }));

  // Export each region’s results as a separate CSV
  Export.table.toDrive({
    collection:     metrics,
    description:    'NDVI_MonthlyMetrics_' + adm2_code,
    fileNamePrefix: adm2_code + '_NDVI_Monthly_full',
    folder:         'EarthEngineExports/NDVI_TIMOR_ADM2',
    fileFormat:     'CSV'
  });
});
