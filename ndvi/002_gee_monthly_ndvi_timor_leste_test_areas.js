// Full GEE script: Extract mean & max monthly NDVI & EVI (Jan 2019–Dec 2025)
// over your Maliana rice agricultural mask, exporting one CSV named by the asset.

// 0. Inputs & Settings
var assetId       = 
    'projects/ee-spenson/assets/food-security-timor-leste/maliana_rice_agricultural_area'
//    'projects/ee-spenson/assets/food-security-timor-leste/triloka_acid_soil_area'
//    'projects/ee-spenson/assets/food-security-timor-leste/caibada_alkaline_soil_area'
//    'projects/ee-spenson/assets/food-security-timor-leste/darasula_research_station_area'
//    'projects/ee-spenson/assets/food-security-timor-leste/fatumaca_research_station_area'
//    'projects/ee-spenson/assets/food-security-timor-leste/natarbora_neutral_soil_area'
;

var assetName     = assetId.split('/').pop();  // "maliana_rice_agricultural_area"
var croplandMaskFC = ee.FeatureCollection(assetId);

var START_YEAR = 2019;
var END_YEAR   = 2025;

// 1. Load & cloud-mask Sentinel-2 SR (bands B8, B4, B2 for NDVI & EVI)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .map(function(img) {
    var scl = img.select('SCL');
    // keep only clear pixels
    return img.updateMask(scl.lt(7));
  })
  .select(['B8','B4','B2']);

// 2. Build server-side list of month-start dates
var firstOfJan = ee.Date.fromYMD(START_YEAR, 1, 1);
var monthCount = ee.Date.fromYMD(END_YEAR, 12, 1)
  .difference(firstOfJan, 'month')
  .add(1);
var monthDates = ee.List.sequence(0, monthCount.subtract(1))
  .map(function(n) { return firstOfJan.advance(n, 'month'); });

// 3. Compute monthly metrics over the entire asset geometry
var regionGeom = croplandMaskFC.geometry();
var metrics = ee.FeatureCollection(
  monthDates.map(function(d) {
    d = ee.Date(d);
    var start   = d;
    var end     = d.advance(1, 'month').advance(-1, 'second');
    var dateStr = d.format('YYYY-MM-dd');

    // Filter & mask for this month & region
    var col = s2Sr.filterDate(start, end).filterBounds(regionGeom);
    var count = col.size();

    // Compute NDVI and EVI stacks
    var ndviCol = col.map(function(img) {
      return img.normalizedDifference(['B8','B4']).rename('NDVI');
    });
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

    // Safely compute each metric (null if no images)
    var meanNdvi = ee.Algorithms.If(
      count.gt(0),
      ndviCol.mean()
        .reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  regionGeom,
          scale:     10,
          maxPixels: 1e13,
          tileScale: 4
        }).get('NDVI'),
      null
    );
    var maxNdvi = ee.Algorithms.If(
      count.gt(0),
      ndviCol.max()
        .reduceRegion({
          reducer:   ee.Reducer.max(),
          geometry:  regionGeom,
          scale:     10,
          maxPixels: 1e13,
          tileScale: 4
        }).get('NDVI'),
      null
    );
    var meanEvi = ee.Algorithms.If(
      count.gt(0),
      eviCol.mean()
        .reduceRegion({
          reducer:   ee.Reducer.mean(),
          geometry:  regionGeom,
          scale:     10,
          maxPixels: 1e13,
          tileScale: 4
        }).get('EVI'),
      null
    );
    var maxEvi = ee.Algorithms.If(
      count.gt(0),
      eviCol.max()
        .reduceRegion({
          reducer:   ee.Reducer.max(),
          geometry:  regionGeom,
          scale:     10,
          maxPixels: 1e13,
          tileScale: 4
        }).get('EVI'),
      null
    );

    // Return a feature with exactly the desired properties
    return ee.Feature(null, {
      Name:         assetName,
      date:         dateStr,
      count_images: count,
      mean_NDVI:    meanNdvi,
      max_NDVI:     maxNdvi,
      mean_EVI:     meanEvi,
      max_EVI:      maxEvi
    });
  })
);

// 4. Export to Drive with assetName in the task description & filename
Export.table.toDrive({
  collection:     metrics,
  description:    'Metrics_' + assetName,
  fileNamePrefix: assetName + '_NDVI_EVI_monthly',
  folder:         'EarthEngineExports_NDVI_EVI',
  fileFormat:     'CSV',
  selectors: [
    'Name',
    'date',
    'count_images',
    'mean_NDVI',
    'max_NDVI',
    'mean_EVI',
    'max_EVI'
  ]
});
