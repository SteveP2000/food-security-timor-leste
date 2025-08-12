// ---------------------------------------------------------------------
// Monthly NDVI/EVI per ADM2, aligned with Script 3 masking & structure
// - s2cloudless + SCL masking
// - NDVI/EVI computed from S2_SR (scaled by 0.0001)
// - Per-month QA: mean/max clear fraction across images
// - One CSV per ADM2 (2019–2025)
// ---------------------------------------------------------------------

// 0) Inputs & Settings
var adm2FC = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/LULC_Mean_Probability_Harvest_cropland_mask_adm2'
);
var START_YEAR = 2019;
var END_YEAR   = 2025;
var CLD_PROB_THR = 40; // consistent with Script 3

// 1) Sentinel-2: join S2_SR with s2cloudless and build CLEAR mask via SCL + prob
var s2sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR, 12, 31).advance(1, 'day'));

var s2prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR, 12, 31).advance(1, 'day'));

// Join on granule ID (system:index works reliably for S2_SR ↔ s2cloudless)
var joined = ee.ImageCollection(ee.Join.saveFirst('clouds').apply({
  primary: s2sr,
  secondary: s2prob,
  condition: ee.Filter.equals({
    leftField: 'system:index',
    rightField: 'system:index'
  })
}));

function maskS2(img) {
  var scl = img.select('SCL');
  var cld = ee.Image(img.get('clouds')).select('probability').rename('MSK_CLDPRB');

  // Keep SCL 4 (veg), 5 (bare), 6 (water); optionally include 11 (snow/ice)
  var keepSCL  = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11));
  var keepProb = cld.lt(CLD_PROB_THR);

  // Apply masks to all bands
  var masked = img.updateMask(keepSCL).updateMask(keepProb);

  // CLEAR = 1 where valid (unmasked), masked elsewhere
  var clear = ee.Image(1).updateMask(masked.mask().reduce(ee.Reducer.min())).rename('CLEAR');

  return masked.addBands(clear);
}

var s2 = joined
  .map(function(img){
    // Ensure SCL band is present; keep only needed bands + CLEAR
    return img.addBands(img.select(['B8','B4','B2','SCL']), null, true);
  })
  .map(maskS2)
  .select(['B8','B4','B2','CLEAR']); // we need these for indices + QA

// 2) Helpers: monthly dates + indices
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var monthCount = ee.Date.fromYMD(END_YEAR, 12, 1).difference(firstOfJan, 'month').add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) { return firstOfJan.advance(n, 'month'); });

function ndviFrom(img) {
  var nir = img.select('B8').multiply(0.0001);
  var red = img.select('B4').multiply(0.0001);
  return nir.subtract(red).divide(nir.add(red)).rename('NDVI');
}
function eviFrom(img) {
  var nir  = img.select('B8').multiply(0.0001);
  var red  = img.select('B4').multiply(0.0001);
  var blue = img.select('B2').multiply(0.0001);
  return nir.subtract(red).multiply(2.5)
            .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
            .rename('EVI');
}

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

        // Monthly S2 subset over region
        var col = s2.filterDate(start, end).filterBounds(regionGeom);
        var count = col.size();

        // Attach per-image CLEAR fraction over the region (unmask 0 → includes cloudy area)
        var colWithCF = col.map(function(img) {
          var cf = img.select('CLEAR').unmask(0).reduceRegion({
            reducer:  ee.Reducer.mean(),
            geometry: regionGeom,
            scale:    20,
            maxPixels:1e13,
            tileScale:4
          }).get('CLEAR');
          return img.set('clearFrac', cf);
        });

        // Build NDVI/EVI image collections
        var ndIC = col.map(ndviFrom);
        var evIC = col.map(eviFrom);

        // Monthly composites
        var ndviMeanImg = ndIC.mean();
        var ndviMaxImg  = ndIC.max();
        var eviMeanImg  = evIC.mean();
        var eviMaxImg   = evIC.max();

        // Reductions over region (null if no images)
        var meanNdvi = ee.Algorithms.If(
          count.gt(0),
          ndviMeanImg.reduceRegion({
            reducer: ee.Reducer.mean(),
            geometry: regionGeom,
            scale: 10, maxPixels:1e13, tileScale:4
          }).get('NDVI'),
          null
        );
        var maxNdvi = ee.Algorithms.If(
          count.gt(0),
          ndviMaxImg.reduceRegion({
            reducer: ee.Reducer.max(),
            geometry: regionGeom,
            scale: 10, maxPixels:1e13, tileScale:4
          }).get('NDVI'),
          null
        );
        var meanEvi = ee.Algorithms.If(
          count.gt(0),
          eviMeanImg.reduceRegion({
            reducer: ee.Reducer.mean(),
            geometry: regionGeom,
            scale: 10, maxPixels:1e13, tileScale:4
          }).get('EVI'),
          null
        );
        var maxEvi = ee.Algorithms.If(
          count.gt(0),
          eviMaxImg.reduceRegion({
            reducer: ee.Reducer.max(),
            geometry: regionGeom,
            scale: 10, maxPixels:1e13, tileScale:4
          }).get('EVI'),
          null
        );

        // Clear-sky coverage QA
        var clearFracMean = ee.Algorithms.If(
          count.gt(0), colWithCF.aggregate_mean('clearFrac'), null
        );
        var clearFracMax = ee.Algorithms.If(
          count.gt(0), colWithCF.aggregate_max('clearFrac'), null
        );

        return ee.Feature(null, {
          ADM2_PCODE:      adm2_code,
          date:            dateStr,      // first of month
          count_images:    count,
          clear_frac_mean: clearFracMean, // 0..1 (avg fraction of region that was clear)
          clear_frac_max:  clearFracMax,  // 0..1 (best single-scene coverage)
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
