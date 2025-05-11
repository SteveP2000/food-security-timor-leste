// Load custom ADM2 shapefile for Timor-Leste
var timor_leste_adm2 = ee.FeatureCollection('projects/ee-spenson/assets/tls_admn_ad2_py_s2_unocha_pp');

// Center map on Timor-Leste
Map.centerObject(timor_leste_adm2, 8);

// Define cropland mask function using USGS GFSAD1000
function getCroplandMask(adm2) {
  var cropland = ee.Image('USGS/GFSAD1000_V1').select('landcover').clip(adm2);
  return cropland.eq(1).or(cropland.eq(2)).or(cropland.eq(3))
                 .or(cropland.eq(4)).or(cropland.eq(5));
}

// Calculate cropland ratio for a single ADM2
function calculateCroplandRatio(adm2) {
  var cropland_mask = getCroplandMask(adm2);

  var cropland_area = cropland_mask.multiply(ee.Image.pixelArea())
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: adm2.geometry(),
      scale: 30,
      maxPixels: 1e9
    }).get('landcover');

  var total_area = ee.Image.pixelArea()
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: adm2.geometry(),
      scale: 30,
      maxPixels: 1e9
    }).get('area');

  var cropland_ratio = ee.Number(cropland_area).divide(ee.Number(total_area)).multiply(100);
  return cropland_ratio;
}

// Apply cropland ratio calculation to all ADM2 regions
function calculateCroplandRatiosForADM2() {
  return timor_leste_adm2.map(function(adm2) {
    var ratio = calculateCroplandRatio(adm2);
    return adm2.set('cropland_ratio', ratio);
  });
}

// Compute cropland ratios
var cropland_ratios = calculateCroplandRatiosForADM2();

// Print result to the Console
print('Cropland Ratios by ADM2', cropland_ratios);

// Visualization parameters
var visParams = {
  min: 0,
  max: 100,
  palette: ['ffffff', 'a1dab4', '41b6c4', '2c7fb8', '253494']
};

// Style and add to map
Map.addLayer(
  cropland_ratios.style({
    color: '000000',
    fillColor: {
      property: 'cropland_ratio',
      mode: 'linear',
      min: 0,
      max: 100,
      palette: visParams.palette
    },
    width: 1
  }),
  {},
  'Cropland Ratio (%)'
);

// Optional: export to Drive
Export.table.toDrive({
  collection: cropland_ratios,
  description: 'timor_adm2_cropland_ratios',
  folder: 'GEE_Cropland_Ratios_Timor',
  fileFormat: 'CSV',
  selectors: ['ADM2_PCODE', 'ADM1_PCODE', 'cropland_ratio']
});
