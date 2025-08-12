// Full GEE script: Probabilistic “highest‐occurrence” Dynamic World classification,
// plus per-polygon CSV of class areas and GeoTIFF of the resulting label map.

// 0. —– Prerequisite —–
// • Upload your multipolygon asset as an EE FeatureCollection.
// • Replace the Asset ID below if needed.
var regions = ee.FeatureCollection(
//  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp_aileu_vila'
  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp'
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

// 4. Sum up the per‐image class probabilities over the entire period
var sumProb = dwCol
  .select(CLASS_NAMES)
  .reduce(ee.Reducer.sum());

// 5. Find, for each pixel, the class index with the highest summed probability
var probArray = sumProb.toArray();           // [9] array per pixel
var argmaxIdx = probArray.arrayArgmax();     // index of max along axis 0
var dwProbMax = argmaxIdx
  .arrayGet([0])                             // turn [1]→scalar
  .rename('label');

// 6. Visualization palette
var VIS_PALETTE = [
  '377EB8','4DAF4A','A6D854','984EA3','FF7F00',
  'FFFF33','E41A1C','A65628','999999'
];

// 7. Display global result & boundaries
Map.centerObject(regions, 8);
Map.addLayer(
  dwProbMax,
  {min: 0, max: 8, palette: VIS_PALETTE},
  'DW probabilistic max-sum label'
);
Map.addLayer(
  regions.style({color: 'black', fillColor: '00000000'}),
  {},
  'All polygons'
);

// 8. Prepare an Image for area calculations: pixel area + label
var areaImage = ee.Image.pixelArea().addBands(dwProbMax);

// 9. Iterate over each polygon and export
var regionList = regions.toList(regions.size());
var count = regionList.size().getInfo();

for (var i = 0; i < count; i++) {
  // Client-side retrieval
  var feat = ee.Feature(regionList.get(i));
  // Replace 'ADM2_PCODE' with your unique ID field if different
  var id   = feat.get('ADM2_PCODE').getInfo();
  var geom = feat.geometry();
  
  // 9a. Compute per‐class area within this polygon
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
  
  // 9b. Export table to Google Drive
  Export.table.toDrive({
    collection:     areaFC,
    description:    'DW_probMax_area_' + id,
    fileNamePrefix: 'DW_probMax_area_' + id,
    folder:         'EarthEngineExports',
    fileFormat:     'CSV'
  });
  
  // 9c. Export the clipped label mosaic as GeoTIFF
  var dwClipped = dwProbMax.clip(geom);
  Export.image.toDrive({
    image:          dwClipped,
    description:    'DW_probMax_label_' + id,
    fileNamePrefix: 'DW_probMax_label_' + id,
    folder:         'EarthEngineExports',
    region:         geom,
    scale:          10,
    crs:            'EPSG:4326',
    fileFormat:     'GeoTIFF',
    maxPixels:      1e13
  });
}
