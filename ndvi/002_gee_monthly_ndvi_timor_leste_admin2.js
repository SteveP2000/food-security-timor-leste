// ---------------------------------------------------------------------
// Monthly NDVI/EVI per ADM2 (FAST VERSION)
// - Harmonized S2_SR + s2cloudless (inner join)
// - Precompute NDVI/EVI once; monthly CLEAR QA from temporal reducers
// - Single reduceRegion per month for NDVI/EVI and one for CLEAR QA
// ---------------------------------------------------------------------

// 0) Inputs & Settings
var adm2FC = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/LULC_Mean_Probability_Harvest_cropland_mask_adm2_v2'
//  'projects/ee-spenson/assets/food-security-timor-leste/LULC_Impact_Observatory_cropland_mask_adm2'  
);
var START_YEAR = 2019;
var END_YEAR   = 2025;
var CLD_PROB_THR = 40;
var REDUCE_SCALE = 20;

// 1) Sentinel-2: robust inner join (every image has cloud prob)
var s2harmSR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1),
              ee.Date.fromYMD(END_YEAR,   12, 31).advance(1, 'day'));

var s2prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1),
              ee.Date.fromYMD(END_YEAR,   12, 31).advance(1, 'day'));

var innerJoined = ee.Join.inner().apply({
  primary: s2harmSR,
  secondary: s2prob,
  condition: ee.Filter.equals({
    leftField:  'system:index',
    rightField: 'system:index'
  })
});

// Attach MSK_CLDPRB and keep only needed bands early
var withCloudProb = ee.ImageCollection(innerJoined.map(function(pair) {
  var s2  = ee.Image(pair.get('primary'));
  var cp  = ee.Image(pair.get('secondary')).select('probability').rename('MSK_CLDPRB');
  return s2.addBands(cp).select(['B8','B4','B2','SCL','MSK_CLDPRB']);
}));

// Mask function
function maskS2(img) {
  var scl = img.select('SCL');
  var clearSCL  = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11)); // veg, bare, water, snow/ice
  var clearProb = img.select('MSK_CLDPRB').lt(CLD_PROB_THR);
  var masked = img.updateMask(clearSCL).updateMask(clearProb);

  // CLEAR = 1 where valid; masked elsewhere
  var clear = ee.Image(1).updateMask(masked.mask().reduce(ee.Reducer.min())).rename('CLEAR');
  return masked.addBands(clear);
}

// Precompute NDVI/EVI once
function addIndices(img) {
  var nir  = img.select('B8').multiply(0.0001);
  var red  = img.select('B4').multiply(0.0001);
  var blue = img.select('B2').multiply(0.0001);
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  var evi  = nir.subtract(red).multiply(2.5)
               .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
               .rename('EVI');
  return img.addBands([ndvi, evi]);
}

// Final working collection: NDVI, EVI, CLEAR only
var S2 = withCloudProb.map(maskS2).map(addIndices).select(['NDVI','EVI','CLEAR']);

// 2) Helpers: monthly dates
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var monthCount = ee.Date.fromYMD(END_YEAR, 12, 1)
                  .difference(firstOfJan, 'month').add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) { return firstOfJan.advance(n, 'month'); });

// 3) Per-ADM2 processing & export (batched via evaluate)
adm2FC.aggregate_array('ADM2_PCODE').evaluate(function(codes) {
  if (!codes || !codes.length) { print('No ADM2_PCODE values found.'); return; }
  print('Preparing', codes.length, 'ADM2 monthly exports… Open the Tasks tab.');

  codes.forEach(function(adm2_code) {
    var feat       = adm2FC.filter(ee.Filter.eq('ADM2_PCODE', adm2_code)).first();
    var regionGeom = ee.Feature(feat).geometry();

    var metrics = ee.FeatureCollection(
      monthDates.map(function(d0) {
        var d = ee.Date(d0);
        var start   = d;
        var end     = d.advance(1, 'month');
        var dateStr = d.format('YYYY-MM-dd');

        // Monthly subset
        var col   = S2.filterDate(start, end).filterBounds(regionGeom);
        var count = col.size();

        // ---- NDVI/EVI: build 4-band monthly image, reduce once
        var monthlyImg = ee.Image.cat(
          col.select('NDVI').mean().rename('mean_NDVI'),
          col.select('NDVI').max() .rename('max_NDVI'),
          col.select('EVI') .mean().rename('mean_EVI'),
          col.select('EVI') .max() .rename('max_EVI')
        );

        var statsDict = monthlyImg.reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  regionGeom,
          scale:     REDUCE_SCALE,
          maxPixels: 1e13,
          tileScale: 4
        });

        // ---- CLEAR QA from temporal reducers (no per-image loops)
        var clearImg = ee.Image.cat(
          col.select('CLEAR').mean().unmask(0).rename('clear_frac_mean'),
          col.select('CLEAR').max() .unmask(0).rename('clear_frac_max')
        );

        var clearDict = clearImg.reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  regionGeom,
          scale:     REDUCE_SCALE,
          maxPixels: 1e13,
          tileScale: 4
        });

        // Pull values (null if no images)
        var meanNdvi = ee.Algorithms.If(count.gt(0), statsDict.get('mean_NDVI'), null);
        var maxNdvi  = ee.Algorithms.If(count.gt(0), statsDict.get('max_NDVI'),  null);
        var meanEvi  = ee.Algorithms.If(count.gt(0), statsDict.get('mean_EVI'),  null);
        var maxEvi   = ee.Algorithms.If(count.gt(0), statsDict.get('max_EVI'),   null);

        var clearFracMean = ee.Algorithms.If(count.gt(0), clearDict.get('clear_frac_mean'), null);
        var clearFracMax  = ee.Algorithms.If(count.gt(0), clearDict.get('clear_frac_max'),  null);

        return ee.Feature(null, {
          ADM2_PCODE:      adm2_code,
          date:            dateStr,
          count_images:    count,
          clear_frac_mean: clearFracMean, // 0..1
          clear_frac_max:  clearFracMax,  // 0..1
          mean_NDVI:       meanNdvi,
          max_NDVI:        maxNdvi,
          mean_EVI:        meanEvi,
          max_EVI:         maxEvi
        });
      })
    );

    // Export one CSV per ADM2
    Export.table.toDrive({
      collection:     metrics,
      description:    'Metrics_admn2_' + adm2_code,
      fileNamePrefix: adm2_code + '_NDVI_EVI_monthly_admn2',
      folder:         'EarthEngineExports_NDVI_EVI',
      fileFormat:     'CSV',
      selectors: [
        'ADM2_PCODE',
        'date',
        'count_images',
        'clear_frac_mean',
        'clear_frac_max',
        'mean_NDVI',
        'max_NDVI',
        'mean_EVI',
        'max_EVI'
      ]
    });
  });
});
