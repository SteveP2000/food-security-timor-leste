// Harvest‐season (Apr–Jul) Mean‐Probability Argmax DW classification per AOI polygon,
// with s2cloudless + SCL per-pixel masking applied to the matching DW image (saveBest join).

// ------------------ 0) Assets ------------------
var regions = ee.FeatureCollection(
//  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp_aileu_vila'
  'projects/ee-spenson/assets/food-security-timor-leste/tls_admn_ad2_py_s2_unocha_pp'
);

// ------------------ 1) Parameters ------------------
var HARVEST_START = 4;  // April
var HARVEST_END   = 7;  // July
var harvestMonths = ee.List.sequence(HARVEST_START, HARVEST_END);

var START = ee.Date('2019-01-01');
var END   = ee.Date('2025-12-31');

var CLD_PROB_THR = 40;  // s2cloudless threshold (0..100)

// ------------------ 2) Class names & palette ------------------
var CLASS_NAMES = [
  'water','trees','grass','flooded_vegetation',
  'crops','shrub_and_scrub','built','bare','snow_and_ice'
];
var VIS_PALETTE = [
  '377EB8','4DAF4A','A6D854','984EA3','FF7F00',
  'FFFF33','E41A1C','A65628','999999'
];

// ------------------ 3) Sentinel-2 with s2cloudless + SCL ------------------
var s2sr = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterDate(START, END)
  .filterBounds(regions);

var s2prob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterDate(START, END)
  .filterBounds(regions);

// Join S2_SR to s2cloudless by granule id (system:index)
var joined = ee.ImageCollection(ee.Join.saveFirst('clouds').apply({
  primary: s2sr,
  secondary: s2prob,
  condition: ee.Filter.equals({
    leftField: 'system:index',
    rightField: 'system:index'
  })
}));

// Build CLEAR mask using SCL + cloud probability
function maskS2(img) {
  var scl = img.select('SCL');
  var cld = ee.Image(img.get('clouds')).select('probability').rename('MSK_CLDPRB');

  // Keep vegetation (4), bare soil (5), water (6), optionally snow/ice (11)
  var keepSCL   = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11));
  var keepProb  = cld.lt(CLD_PROB_THR);

  // Apply both masks to form a per-pixel CLEAR mask (1 where clear)
  var masked    = img.updateMask(keepSCL).updateMask(keepProb);
  var clearMask = ee.Image(1).updateMask(masked.mask().reduce(ee.Reducer.min()))
                 .rename('CLEAR');

  return img.addBands(clearMask)
            .set('month', ee.Date(img.get('system:time_start')).get('month'));
}

// Final S2 collection: we only need the CLEAR band + month property
var s2Col = joined.map(maskS2).select(['CLEAR']);

// ------------------ 4) Dynamic World (tag month) ------------------
var dwColRaw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate(START, END)
  .filterBounds(regions)
  .map(function(img) {
    var m = ee.Date(img.get('system:time_start')).get('month');
    return img.set('month', m);
  });

// ------------------ 5) DW masked by best-matching S2 CLEAR (saveBest) ------------------
function dwMaskedByS2Clear(geom) {
  var s2Harvest = s2Col
    .filterBounds(geom)
    .filter(ee.Filter.inList('month', harvestMonths));

  var dwHarvest = dwColRaw
    .filterBounds(geom)
    .filter(ee.Filter.inList('month', harvestMonths));

  // Allow small time differences and require spatial overlap
  var temporal = ee.Filter.maxDifference({
    difference: 3 * 60 * 60 * 1000, // 3 hours
    leftField:  'system:time_start',
    rightField: 'system:time_start'
  });
  var spatial = ee.Filter.intersects({ leftField: '.geo', rightField: '.geo' });
  var both    = ee.Filter.and(temporal, spatial);

  // Attach best-matching S2 CLEAR to each DW image
  var saveBest = ee.Join.saveBest({ matchKey: 's2', measureKey: 'timeDiff' });
  var dwWithS2 = ee.ImageCollection(saveBest.apply(dwHarvest, s2Harvest, both));

  // Mask each DW image by the CLEAR band from its best S2
  return ee.ImageCollection(dwWithS2.map(function(img) {
    img = ee.Image(img);
    var s2   = ee.Image(img.get('s2'));
    var clear = ee.Algorithms.If(
      s2, ee.Image(s2).select('CLEAR'),
      ee.Image(0).selfMask() // no match → empty mask
    );
    return img.select(CLASS_NAMES)
              .updateMask(ee.Image(clear))
              .copyProperties(img, img.propertyNames());
  }));
}

// ------------------ 6) Iterate over polygons ------------------
var regionList = regions.toList(regions.size());
var n = regionList.size().getInfo();

for (var i = 0; i < n; i++) {
  var feat = ee.Feature(regionList.get(i));
  var code = feat.get('ADM2_PCODE').getInfo();
  if (!code) { print('⚠️ Skipping feature #' + i + ' with null ADM2_PCODE'); continue; }
  var geom = feat.geometry();

  // (a) Per-pixel masked DW (harvest months only)
  var dwMasked = dwMaskedByS2Clear(geom);

  // (b) Count DW images after masking (for QA)
  var imgCount = dwMasked.size();
  print(code, 'harvest DW images (after mask):', imgCount);

  // (c) Mean-probability across harvest months (per pixel)
  var countImg = dwMasked.select('water').count();                // valid obs count
  var sumProb  = dwMasked.select(CLASS_NAMES).reduce(ee.Reducer.sum());
  var meanProb = sumProb.divide(countImg.max(1))                  // avoid div/0
                        .updateMask(countImg.gt(0));              // ensure only pixels with ≥1 obs

  // (d) Argmax → label 0..8
  var labelMap = meanProb
    .toArray()
    .arrayArgmax()
    .arrayGet([0])
    .rename('label')
    .clip(geom);

  // (e) Conservative gap fill (3 px, ~30 m)
  var filled = labelMap.unmask(
                  labelMap.focal_mode({radius: 3, units: 'pixels', kernelType: 'square'})
               ).clip(geom);

  // (f) Class areas (m² → ha)
  var areaImg = ee.Image.pixelArea().addBands(filled);
  var areaDict = areaImg.reduceRegion({
    reducer:   ee.Reducer.sum().group({groupField:1, groupName:'label'}),
    geometry:  geom,
    scale:     10,
    maxPixels: 1e13,
    tileScale: 4
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

  // (g) Export CSV (class areas)
  Export.table.toDrive({
    collection:     areaFC,
    description:    'Harvest_DW_meanProb_area_'  + code,
    fileNamePrefix: 'Harvest_DW_meanProb_area_'  + code,
    folder:         'EarthEngineExports',
    fileFormat:     'CSV'
  });

  // (h) Export GeoTIFF (label map) — use native projection; don't force EPSG:4326
  Export.image.toDrive({
    image:          filled,
    description:    'Harvest_DW_meanProb_label_' + code,
    fileNamePrefix: 'Harvest_DW_meanProb_label_' + code,
    folder:         'EarthEngineExports',
    region:         geom,
    scale:          10,
    fileFormat:     'GeoTIFF',
    maxPixels:      1e13
  });

  // (i) QA layer
  Map.addLayer(filled, {min:0, max:8, palette:VIS_PALETTE}, 'DW ' + code);
}

// Center map on all AOIs
Map.centerObject(regions, 9);
