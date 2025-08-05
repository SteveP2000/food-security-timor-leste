// Full GEE script: process each of 65 polygons separately,
// exporting per-polygon CSV of DW class areas and GeoTIFF of latest DW labels,
// plus visualizing the global mosaic and AOI boundaries.

// 0. —– Prerequisite —–
// • Upload your multipolygon asset as an EE FeatureCollection.
// • Replace this Asset ID if yours is different.
var regions = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp_aileu_vila'
);

// 1. Define the full time window
var GLOBAL_START = ee.Date('2019-01-01');
var GLOBAL_END   = ee.Date('2025-12-31');

// 2. Load and filter the DW collection over the entire region
var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate(GLOBAL_START, GLOBAL_END)
  .filterBounds(regions);

// 3. Build a global “latest-pixel” mosaic of the DW label band
var dwMosaicGlobal = dwCol
  .sort('system:time_start', false)  // most recent first
  .mosaic();

// 4. Prepare an Image for area calculations: pixelArea + label band
var areaImage = ee.Image.pixelArea()
  .addBands(dwMosaicGlobal.select('label'));

// 5. Visualization palette for Dynamic World
var VIS_PALETTE = [
  '419bdf','397d49','88b053','7a87c6','e49635',
  'dfc35a','c4281b','a59b8f','b39fe1'
];

// 6. Display global mosaic & all region boundaries
Map.centerObject(regions, 8);
Map.addLayer(
  dwMosaicGlobal.select('label'),
  {min: 0, max: 8, palette: VIS_PALETTE},
  'DW latest-pixel mosaic (all regions)'
);
Map.addLayer(
  regions.style({color: 'black', fillColor: '00000000'}),
  {},
  'All polygons'
);

// 7. Export per-feature results
//    We convert the FC to a client-side list to iterate.
var regionList = regions.toList(regions.size());
var count = regionList.size().getInfo();

for (var i = 0; i < count; i++) {
  // Client-side: grab the i-th feature
  var feat   = ee.Feature(regionList.get(i));
  // Get a unique ID for the feature. Replace 'ADM2_PCODE' with your ID field.
  var id     = feat.get('ADM2_PCODE').getInfo();
  var geom   = feat.geometry();
  
  // 7a. Compute per-class area within this feature
  var areaDict = areaImage.reduceRegion({
    reducer: ee.Reducer.sum().group({
      groupField: 1,
      groupName:  'label'
    }),
    geometry:   geom,
    scale:      10,
    maxPixels:  1e13
  });
  
  var classAreas = ee.List(areaDict.get('groups'));
  var areaFC = ee.FeatureCollection(classAreas.map(function(item) {
    var d = ee.Dictionary(item);
    var m2 = ee.Number(d.get('sum'));
    return ee.Feature(null, {
      polygon_id: id,
      class:      d.get('label'),
      area_m2:    m2,
      area_ha:    m2.divide(1e4)
    });
  }));
  
  // 7b. Export the table to Drive
  Export.table.toDrive({
    collection:     areaFC,
    description:    'DW_area_' + id,
    fileNamePrefix: 'DW_area_' + id,
    folder:         'EarthEngineExports',
    fileFormat:     'CSV'
  });
  
  // 7c. Export the clipped label mosaic to Drive
  var dwClipped = dwMosaicGlobal.select('label').clip(geom);
  Export.image.toDrive({
    image:          dwClipped,
    description:    'DW_label_' + id,
    fileNamePrefix: 'DW_label_' + id,
    folder:         'EarthEngineExports',
    region:         geom,
    scale:          10,
    crs:            'EPSG:4326',
    fileFormat:     'GeoTIFF',
    maxPixels:      1e13
  });
}
