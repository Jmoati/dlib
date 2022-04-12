<?php

use Symfony\Component\Finder\Finder;

if (!extension_loaded("pdlib"))  die('skip pdlib extension missing');
if (!function_exists("bzopen"))  die('skip bz2 extension missing');

require __DIR__ . '/vendor/autoload.php';

$models = [
    "detection" => [
        "uri"=>"http://dlib.net/files/mmod_human_face_detector.dat.bz2",
    ],
    "prediction" => [
        "uri"=>"http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
    ],
    "recognition" => [
        "uri"=>"http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    ],
];

foreach ($models as $modelName => $modelBag) {
    printf("Processing %s model\n", $modelName);
    $bz2_filename = array_values(array_slice(explode("/", $modelBag["uri"]), -1))[0];
    $temp_bz2_file = sys_get_temp_dir() . "/" . $bz2_filename;
    $dat_filename = array_values(array_slice(explode(".", $bz2_filename), 0))[0] . ".dat";
    $temp_dat_file = __DIR__ . "/models/" . $dat_filename;
    
    $models[$modelName]["local_path"] = $temp_dat_file;
    
    if (file_exists($temp_dat_file)) {
        continue;
    }
    
    file_put_contents($temp_bz2_file, fopen($modelBag["uri"], 'r'));
    $bz = bzopen($temp_bz2_file, "r");
    $decompressed_file = "";
    while (!feof($bz)) {
        $decompressed_file .= bzread($bz, 4096);
    }
    bzclose($bz);
    
    file_put_contents($temp_dat_file, $decompressed_file);
}

$fd = new CnnFaceDetection($models["detection"]["local_path"]);
$ld = new FaceLandmarkDetection($models["prediction"]["local_path"]);
$fr = new FaceRecognition($models["recognition"]["local_path"]);

$faces = [];

foreach (Finder::create()->in(__DIR__ . '/images/')->files() as $file) {
    $filename = $file->getRealPath();
    $jpeg = __DIR__.'/jpeg/'.$file->getBasename() . '.jpg';
    
    printf("Converting %s to jpeg\n", $file->getBasename());
    
    $im = new Imagick($filename);
    $im->setImageCompressionQuality(100);
    $im = $im->mergeImageLayers(\Imagick::LAYERMETHOD_FLATTEN);
    $im->setInterlaceScheme(Imagick::INTERLACE_PLANE);
    $im->stripImage();
    $im->writeImage($jpeg);
    
    printf("Detecting faces ...\n");
    $detected_faces = $fd->detect($jpeg);

    foreach ($detected_faces as $offset => $face) {
        printf("Analyzing face #%s\n", $offset);
        $source = new Imagick($jpeg);
        $source->cropImage($face['right']-$face['left'], $face['bottom']-$face['top'], $face['left'], $face['top']);
        $source->writeImage('./faces/'.count($faces).'.jpeg');
        $landmarks = $ld->detect($jpeg, $face);
        $descriptor = $fr->computeDescriptor($jpeg, $landmarks);
        
        $face['descriptor'] = $descriptor;
        $faces[] = $face;
    }
    
    file_put_contents('data.json', json_encode($faces));
}


$data = json_decode(file_get_contents('data.json'), true);

$edges = [];
for ($i = 0, $l = count($data); $i < $l; ++$i) {
    if ($data[$i]['detection_confidence'] < 0.99) {
        continue;
    }
    
    foreach ($data as $offset => $face) {
        if ($offset <= $i)  {
            continue;
        }
        
        if ($i === $offset || $face['detection_confidence'] < 0.99) {
            continue;
        }
        
        if (dlib_vector_length($data[$i]['descriptor'], $face['descriptor']) < 0.4) {
            $edges[] = [$i, $offset];
        }
    }
}


dd($edges);