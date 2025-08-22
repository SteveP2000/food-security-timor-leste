// ---------------------------------------------------------------------
// Monthly NDVI/EVI for a single AOI (no ADM2_PCODE)
// - AOI: projects/ee-spenson/assets/food-security-timor-leste/seloi_craic_rice_agricultural_area
// - s2cloudless + SCL masking (keep 4/5/6[/11])
// - NDVI/EVI from Harmonized S2_SR (scaled by 0.0001)
// - Per-month QA: mean/max clear fraction across images
// - Single CSV export (2019–2025)
// ---------------------------------------------------------------------

// 0) Inputs & Settings
var ASSET_ID = 
    'projects/ee-spenson/assets/food-security-timor-leste/seloi_craic_rice_agricultural_area'
//  'projects/ee-spenson/assets/food-security-timor-leste/maliana_rice_agricultural_area'
//  'projects/ee-spenson/assets/food-security-timor-leste/triloka_acid_soil_area'
//  'projects/ee-spenson/assets/food-security-timor-leste/caibada_alkaline_soil_area'
//  'projects/ee-spenson/assets/food-security-timor-leste/darasula_research_station_area'
//  'projects/ee-spenson/assets/food-security-timor-leste/fatumaca_research_station_area'
//  'projects/ee-spenson/assets/food-security-timor-leste/natarbora_neutral_soil_area'
  ;
var REGION_NAME = ASSET_ID.split('/').pop();

var aoiFC = ee.FeatureCollection(ASSET_ID);
var regionGeom = aoiFC.geometry();

var START_YEAR = 2019;
var END_YEAR   = 2025;
var CLD_PROB_THR = 40; // consistent with v2/script 3

// 1) Sentinel-2: join Harmonized S2_SR with s2cloudless and build CLEAR mask via SCL + prob
var s2harmSR = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(regionGeom)
  .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR, 12, 31).advance(1, 'day'));

var s2prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterBounds(regionGeom)
  .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR, 12, 31).advance(1, 'day'));

// Join on granule ID (system:index works for S2_SR_HARMONIZED ↔ s2cloudless)
var joined = ee.ImageCollection(ee.Join.saveFirst('clouds').apply({
  primary: s2harmSR,
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

  // Apply masks
  var masked = img.updateMask(keepSCL).updateMask(keepProb);

  // CLEAR = 1 where valid (unmasked), masked elsewhere
  var clear = ee.Image(1).updateMask(masked.mask().reduce(ee.Reducer.min())).rename('CLEAR');

  return masked.addBands(clear);
}

var s2 = joined
  .map(function(img){
    // Ensure the bands exist; keep only what we need plus CLEAR
    return img.addBands(img.select(['B8','B4','B2','SCL']), null, true);
  })
  .map(maskS2)
  .select(['B8','B4','B2','CLEAR']);

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

// 3) Build monthly metrics for the single AOI
var metrics = ee.FeatureCollection(
  monthDates.map(function(d0) {
    var d = ee.Date(d0);
    var start   = d;
    var end     = d.advance(1, 'month');
    var dateStr = d.format('YYYY-MM-dd');

    var col = s2.filterDate(start, end).filterBounds(regionGeom);
    var count = col.size();

    // Per-image CLEAR fraction over the region (unmask 0 to include cloudy area)
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

    // NDVI/EVI monthly composites
    var ndIC = col.map(ndviFrom);
    var evIC = col.map(eviFrom);

    var ndviMeanImg = ndIC.mean();
    var ndviMaxImg  = ndIC.max();
    var eviMeanImg  = evIC.mean();
    var eviMaxImg   = evIC.max();

    // Reductions (null if no images)
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
      region_name:     REGION_NAME,
      date:            dateStr,       // first of month
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

// 4) Export a single CSV (task name uses the asset's base name)
Export.table.toDrive({
  collection:     metrics,
  description:    'Metrics_admn2_' + REGION_NAME, // exactly: Metrics_admn2_seloi_craic_rice_agricultural_area
  fileNamePrefix: REGION_NAME + '_NDVI_EVI_monthly',
  folder:         'EarthEngineExports_NDVI_EVI',
  fileFormat:     'CSV',
  selectors: [
    'region_name',
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

// Optional: quick map cue
Map.centerObject(aoiFC, 12);
Map.addLayer(aoiFC, {color: 'yellow'}, 'AOI');
