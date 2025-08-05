// Full GEE script: “Mean‐Probability Argmax” Dynamic World classification,
// normalized by number of observations, plus per-polygon CSV & GeoTIFF exports.

// 0. —– Prerequisite —–
// • Upload your multipolygon asset as an EE FeatureCollection.
// • Replace the Asset ID below if needed.
var regions = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp_aileu_vila'
);

// 1. Define time window
var GLOBAL_START = ee.Date('2019-01-01');
var GLOBAL_END   = ee.Date('2025-12-31');

// 2. Load Dynamic World over the full region + date range
var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate(GLOBAL_START, GLOBAL_END)
  .filterBounds(regions);

// 3. Class names in DW (order matters)
var CLASS_NAMES = [
  'water', 'trees', 'grass', 'flooded_vegetation',
  'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
];

// 4. Count the number of observations per pixel
//    We can use any single band—for example 'water'
var countImage = dwCol
  .select('water')
  .count()
  .rename('obs_count');

// 5. Sum up the per-image class probabilities over the entire period
var sumProb = dwCol
  .select(CLASS_NAMES)
  .reduce(ee.Reducer.sum());

// 6. Compute mean probability per class
var meanProb = sumProb.divide(countImage);

// 7. Find, for each pixel, the class index with the highest mean probability
var meanArray = meanProb.toArray();           // array of length 9 per pixel
var argmaxIdx = meanArray.arrayArgmax();      // index of max along axis 0
var dwMeanMax = argmaxIdx
  .arrayGet([0])                              // turn [1]→scalar
  .rename('label');

// 8. Visualization palette (colorblind‐friendly)
var VIS_PALETTE = [
  '377EB8','4DAF4A','A6D854','984EA3','FF7F00',
  'FFFF33','E41A1C','A65628','999999'
];

// 9. Display global result & boundaries
Map.centerObject(regions, 8);
Map.addLayer(
  dwMeanMax,
  {min: 0, max: 8, palette: VIS_PALETTE},
  'DW mean-probability argmax'
);
Map.addLayer(
  regions.style({color: 'black', fillColor: '00000000'}),
  {},
  'All polygons'
);

// 10. Prepare an Image for area calculations: pixel area + label
var areaImage = ee.Image.pixelArea().addBands(dwMeanMax);

// 11. Iterate over each polygon and export
var regionList = regions.toList(regions.size());
var count = regionList.size().getInfo();

for (var i = 0; i < count; i++) {
  // Client-side retrieval
  var feat = ee.Feature(regionList.get(i));
  // Replace 'ADM2_PCODE' with your unique ID field if different
  var id   = feat.get('ADM2_PCODE').getInfo();
  var geom = feat.geometry();
  
  // 11a. Compute per-class area within this polygon
  var areaDict = areaImage.reduceRegion({
    reducer: ee.Reducer.sum().group({
      groupField: 1,
      groupName:  'label'
    }),
    geometry:  geom,
    scale:     10,
    maxPixels: 1e13
  });
  
  var classAreas = ee.List(areaDict.get('groups'));
  var areaFC = ee.FeatureCollection(classAreas.map(function(item) {
    var d    = ee.Dictionary(item);
    var m2   = ee.Number(d.get('sum'));
    return ee.Feature(null, {
      polygon_id: id,
      class:      d.get('label'),
      area_m2:    m2,
      area_ha:    m2.divide(1e4)
    });
  }));
  
  // 11b. Export table to Google Drive
  Export.table.toDrive({
    collection:     areaFC,
    description:    'DW_meanProb_area_' + id,
    fileNamePrefix: 'DW_meanProb_area_' + id,
    folder:         'EarthEngineExports',
    fileFormat:     'CSV'
  });
  
  // 11c. Export the clipped label map as GeoTIFF
  var dwClipped = dwMeanMax.clip(geom);
  Export.image.toDrive({
    image:          dwClipped,
    description:    'DW_meanProb_label_' + id,
    fileNamePrefix: 'DW_meanProb_label_' + id,
    folder:         'EarthEngineExports',
    region:         geom,
    scale:          10,
    crs:            'EPSG:4326',
    fileFormat:     'GeoTIFF',
    maxPixels:      1e13
  });
}
