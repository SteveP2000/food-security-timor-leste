// 0. Inputs & Settings
var adm2FC = ee.FeatureCollection(
  //'projects/ee-spenson/assets/food-security-timor-leste/LULC_Mean_Probability_Harvest_cropland_mask_adm2'
  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp'
);
var START_YEAR = 2019;
var END_YEAR   = 2025;

// 1. Load & cloud‐mask Sentinel-2 SR (bands B8, B4, B2)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .map(function(img) {
    return img.updateMask(img.select('SCL').lt(7));
  })
  .select(['B8','B4','B2']);

// 2. Monthly dates
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var monthCount = ee.Date.fromYMD(END_YEAR, 12, 1)
  .difference(firstOfJan, 'month').add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) { return firstOfJan.advance(n, 'month'); });

// 3. Loop per ADM2
adm2FC.aggregate_array('ADM2_PCODE').getInfo().forEach(function(adm2_code) {
  var feat       = adm2FC.filter(ee.Filter.eq('ADM2_PCODE', adm2_code)).first();
  var regionGeom = feat.geometry();

  var metrics = ee.FeatureCollection(
    monthDates.map(function(d) {
      d = ee.Date(d);
      var start   = d;
      var end     = d.advance(1, 'month').advance(-1, 'second');
      var dateStr = d.format('YYYY-MM-dd');
      var col     = s2Sr.filterDate(start, end).filterBounds(regionGeom);
      var count   = col.size();

      // NDVI (0–1)
      var ndviMeanImg = col
        .map(function(img){
          var nir = img.select('B8').multiply(0.0001);
          var red = img.select('B4').multiply(0.0001);
          return nir.subtract(red).divide(nir.add(red)).rename('NDVI');
        })
        .mean();
      var ndviMaxImg = col
        .map(function(img){
          var nir = img.select('B8').multiply(0.0001);
          var red = img.select('B4').multiply(0.0001);
          return nir.subtract(red).divide(nir.add(red)).rename('NDVI');
        })
        .max();

      // EVI (–1…1)
      var eviMeanImg = col
        .map(function(img){
          var nir  = img.select('B8').multiply(0.0001);
          var red  = img.select('B4').multiply(0.0001);
          var blue = img.select('B2').multiply(0.0001);
          return nir.subtract(red).multiply(2.5)
                    .divide(nir.add(red.multiply(6))
                               .subtract(blue.multiply(7.5))
                               .add(1))
                    .rename('EVI');
        })
        .mean();
      var eviMaxImg = col
        .map(function(img){
          var nir  = img.select('B8').multiply(0.0001);
          var red  = img.select('B4').multiply(0.0001);
          var blue = img.select('B2').multiply(0.0001);
          return nir.subtract(red).multiply(2.5)
                    .divide(nir.add(red.multiply(6))
                               .subtract(blue.multiply(7.5))
                               .add(1))
                    .rename('EVI');
        })
        .max();

      // Four explicit reductions
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

      return ee.Feature(null, {
        ADM2_PCODE:   adm2_code,
        date:         dateStr,
        count_images: count,
        mean_NDVI:    meanNdvi,
        max_NDVI:     maxNdvi,
        mean_EVI:     meanEvi,
        max_EVI:      maxEvi
      });
    })
  );

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
      'mean_NDVI',
      'max_NDVI',
      'mean_EVI',
      'max_EVI'
    ]
  });
});
