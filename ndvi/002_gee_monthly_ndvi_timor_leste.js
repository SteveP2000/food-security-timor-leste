// Full GEE script: Extract mean & max monthly NDVI (Jan 2019–Dec 2025) over your Aileu Vila cropland mask,
// with robust handling of months that have no cloud‐free pixels.

// 0. —– Assets & Settings —–
var croplandMaskFC = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/LULC_Mean_Probability_Harvest_cropland_mask_aileu_vila'
);
var regionGeom = croplandMaskFC.geometry();

var START_YEAR = 2019;
var END_YEAR   = 2025;

// 1. Load & cloud‐mask Sentinel-2 SR once (keep only B8 & B4)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(regionGeom)
  .map(function(img) {
    // SCL < 7 = clear
    var clear = img.select('SCL').lt(7);
    return img.updateMask(clear);
  })
  .select(['B8','B4']);

// 2. Build server‐side list of month‐start dates
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var firstOfDec = ee.Date.fromYMD(END_YEAR, 12, 1);
var monthCount = firstOfDec.difference(firstOfJan, 'month').add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) {
    return firstOfJan.advance(n, 'month');
  });

// 3. Map over each month to compute metrics
var metricsFC = ee.FeatureCollection(
  monthDates.map(function(d) {
    d = ee.Date(d);
    var year  = d.get('year');
    var month = d.get('month');
    var start = d;
    var end   = start.advance(1, 'month').advance(-1, 'second');
    
    // Filter to this month and region
    var col = s2Sr
      .filterDate(start, end)
      .filterBounds(regionGeom);
    
    // Compute NDVI per image
    var ndviCol = col.map(function(img) {
      return img.normalizedDifference(['B8','B4']).rename('NDVI');
    });
    
    // Mean & max composites
    var meanImg = ndviCol.mean();
    var maxImg  = ndviCol.max();
    
    // Reduce over the cropland mask
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
    
    // Safely extract NDVI, or null if no pixels
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
    
    // Return a feature for this month
    return ee.Feature(null, {
      year:       year,
      month:      month,
      mean_NDVI:  meanVal,
      max_NDVI:   maxVal
    });
  })
);

// 4. QA: print first few rows
print('Monthly NDVI metrics:', metricsFC.limit(12));

// 5. Export as CSV
Export.table.toDrive({
  collection:     metricsFC,
  description:    'AileuVila_Cropland_NDVI_MonthlyMetrics',
  fileNamePrefix: 'AileuVila_NDVI_Monthly',
  folder:         'EarthEngineExports',
  fileFormat:     'CSV'
});
