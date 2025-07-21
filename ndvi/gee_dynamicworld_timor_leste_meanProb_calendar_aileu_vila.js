// Full GEE script: Harvest‐season (Apr–Jul) Mean‐Probability Argmax DW classification per AOI polygon,
// with cloud masking, exporting one CSV and one GeoTIFF per polygon.

// 0. —– Assets —–
var regions = ee.FeatureCollection(
  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp_aileu_vila'
);

// 1. —– Harvest months to use —–
var HARVEST_START = 4;  // April
var HARVEST_END   = 7;  // July
var harvestMonths = ee.List.sequence(HARVEST_START, HARVEST_END);

// 2. —– Global time window —–
var START = ee.Date('2019-01-01');
var END   = ee.Date('2025-12-31');

// 3. —– Load & prepare Dynamic World —–
var dwCol = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate(START, END)
  .filterBounds(regions)
  .map(function(img) {
    var m = ee.Date(img.get('system:time_start')).get('month');
    return img.set('month', m);
  });

// 4. —– Load & cloud‐mask Sentinel-2 —–
var s2Col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterDate(START, END)
  .filterBounds(regions)
  .map(function(img) {
    var qa = img.select('QA60');
    // Bits 10 and 11 are clouds and cirrus
    var mask = qa.bitwiseAnd(1 << 10).eq(0)
               .and(qa.bitwiseAnd(1 << 11).eq(0));
    return img.updateMask(mask)
              .set('month', ee.Date(img.get('system:time_start')).get('month'));
  });

// 5. —– Class names & visualization palette —–
var CLASS_NAMES = [
  'water','trees','grass','flooded_vegetation',
  'crops','shrub_and_scrub','built','bare','snow_and_ice'
];
var VIS_PALETTE = [
  '377EB8','4DAF4A','A6D854','984EA3','FF7F00',
  'FFFF33','E41A1C','A65628','999999'
];

// 6. —– Iterate over each polygon —–
var regionList = regions.toList(regions.size());
var n = regionList.size().getInfo();

for (var i = 0; i < n; i++) {
  // Client‐side fetch of the i-th feature
  var feat = ee.Feature(regionList.get(i));
  var code = feat.get('ADM2_PCODE').getInfo();
  if (!code) {
    print('⚠️ Skipping feature #' + i + ' with null ADM2_PCODE');
    continue;
  }
  var geom = feat.geometry();
  
  // 6a. Filter Dynamic World to harvest months & this AOI
  var dwHarvest = dwCol
    .filterBounds(geom)
    .filter(ee.Filter.inList('month', harvestMonths));
  
  // 6b. Filter Sentinel-2 to harvest months & this AOI
  var s2Harvest = s2Col
    .filterBounds(geom)
    .filter(ee.Filter.inList('month', harvestMonths));
  
  // 6c. Identify cloud-free dates from S2
  var cfDates = s2Harvest.aggregate_array('system:time_start');
  
  // 6d. Restrict DW to those cloud-free dates
  dwHarvest = dwHarvest.filter(ee.Filter.inList('system:time_start', cfDates));
  
  // 6e. Mean‐Probability Argmax classification
  var countImg = dwHarvest.select('water').count();
  var sumProb  = dwHarvest.select(CLASS_NAMES).reduce(ee.Reducer.sum());
  var meanProb = sumProb.divide(countImg);
  var labelMap = meanProb
    .toArray()
    .arrayArgmax()
    .arrayGet([0])
    .rename('label')
    .clip(geom);
  
  // 6f. Fill small gaps (optional)
  var filled = labelMap.unmask(labelMap.focal_mode(50, 'square', 'pixels'));
  
  // 6g. Compute class areas (m² → ha)
  var areaImg = ee.Image.pixelArea().addBands(filled);
  var areaDict = areaImg.reduceRegion({
    reducer:    ee.Reducer.sum().group({groupField:1, groupName:'label'}),
    geometry:   geom,
    scale:      10,
    maxPixels:  1e13,
    tileScale:  4
  });
  var groups = ee.List(areaDict.get('groups'));
  var areaFC = ee.FeatureCollection(groups.map(function(item) {
    var d  = ee.Dictionary(item);
    var m2 = ee.Number(d.get('sum'));
    return ee.Feature(null, {
      ADM2_PCODE:  code,
      start_month: HARVEST_START,
      end_month:   HARVEST_END,
      class:       d.get('label'),
      area_m2:     m2,
      area_ha:     m2.divide(1e4)
    });
  }));
  
  // 6h. Export class‐area CSV
  Export.table.toDrive({
    collection:     areaFC,
    description:    'Harvest_DW_meanProb_area_'  + code,
    fileNamePrefix: 'Harvest_DW_meanProb_area_'  + code,
    folder:         'EarthEngineExports',
    fileFormat:     'CSV'
  });
  
  // 6i. Export classification GeoTIFF
  Export.image.toDrive({
    image:          filled,
    description:    'Harvest_DW_meanProb_label_' + code,
    fileNamePrefix: 'Harvest_DW_meanProb_label_' + code,
    folder:         'EarthEngineExports',
    region:         geom,
    scale:          10,
    crs:            'EPSG:4326',
    fileFormat:     'GeoTIFF',
    maxPixels:      1e13
  });
  
  // 6j. (Optional) Add to Map for QA
  Map.addLayer(filled, {min:0, max:8, palette:VIS_PALETTE}, 'DW ' + code);
}

// Center map on all AOIs
Map.centerObject(regions, 9);
