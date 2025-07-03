// Full GEE script: Sentinel-2 & Dynamic World over Aileu Vila (2019–2025),
// CSV export of class areas, plus GeoTIFF export of the latest DW land-cover mosaic.

// 0. —– Prerequisite —–
// Upload your Aileu Vila shapefile as an Earth Engine asset and replace this ID:
var region = ee.FeatureCollection('projects/ee-spenson/assets/tls_admn_ad2_py_s2_unocha_pp_aileu_vila');

// 1. Time window
var GLOBAL_START = ee.Date('2019-01-01');
var GLOBAL_END   = ee.Date('2025-12-31');

// 2. Load collections filtered to our AOI + date range
var s2Col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterDate(GLOBAL_START, GLOBAL_END)
  .filterBounds(region)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate(GLOBAL_START, GLOBAL_END)
  .filterBounds(region);

// 3. Build a “latest-pixel” mosaic of the DW label band, clipped to the AOI
var dwMosaic = dwCol
  .sort('system:time_start', false)  // most recent first
  .mosaic()
  .clip(region);

// 4. Visualization palette
var VIS_PALETTE = [
  '419bdf','397d49','88b053','7a87c6','e49635',
  'dfc35a','c4281b','a59b8f','b39fe1'
];

// 5. Display on the map
Map.centerObject(region, 11);

// (a) Sentinel-2 median composite for context
var s2Composite = s2Col.median().clip(region);
Map.addLayer(
  s2Composite,
  {bands: ['B4','B3','B2'], min: 0, max: 3000},
  'Sentinel-2 median RGB'
);

// (b) Dynamic World raw label layer (0–8) with palette
Map.addLayer(
  dwMosaic.select('label'),
  {min: 0, max: 8, palette: VIS_PALETTE},
  'Dynamic World land cover'
);

// (c) AOI boundary
Map.addLayer(
  region.style({color: 'red', fillColor: '00000000'}),
  {},
  'Aileu Vila boundary'
);

// 6. Calculate per-class area (in m²), then convert to hectares
var areaImage = ee.Image.pixelArea()
  .addBands(dwMosaic.select('label'));

var areaDict = areaImage.reduceRegion({
  reducer: ee.Reducer.sum().group({
    groupField: 1,      // grouping on the 'label' band
    groupName: 'label' 
  }),
  geometry: region,
  scale: 10,
  maxPixels: 1e13
});

var classAreas = ee.List(areaDict.get('groups'));
var areaFC = ee.FeatureCollection(classAreas.map(function(item) {
  var dict   = ee.Dictionary(item);
  var area_m2 = dict.get('sum');
  var area_ha = ee.Number(area_m2).divide(1e4);
  return ee.Feature(null, {
    class:   dict.get('label'),
    area_m2: area_m2,
    area_ha: area_ha
  });
}));

print('DW land-cover area per class (ha):', areaFC);

// 7. Export the class-area table as CSV to Google Drive
Export.table.toDrive({
  collection:  areaFC,
  description: 'DW_LULC_area_AileuVila_2019_2025',
  fileFormat:  'CSV'
});

// 8. Export the latest DW label mosaic as a GeoTIFF to Google Drive
Export.image.toDrive({
  image:          dwMosaic.select('label'),
  description:    'DW_LULC_label_AileuVila_2019_2025',
  fileNamePrefix: 'DW_LULC_label_AileuVila',
  folder:         'EarthEngineExports',      // optional: change or remove
  region:         region.geometry(),
  scale:          10,
  crs:            'EPSG:4326',
  fileFormat:     'GeoTIFF',
  maxPixels:      1e13
});
