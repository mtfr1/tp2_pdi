<?php
if(isset($_POST['submit'])){
	if(isset($_POST['tecnica'])){
		$tecnica = $_POST['tecnica'];
	}
	else{
		$tecnica = 'lbp';
	}

	$file = $_FILES['image'];
	$imageName = $file['name'];
	$imageTmp = $file['tmp_name'];
	$scriptFile = "script_web.py";
	
	$imageExt = explode('.', $imageName);
	$imageActualExt = strtolower(end($imageExt));
	
	$imageNameNew = uniqid('', true).'.'.$imageActualExt;
	$imageDest = 'uploaded_images/'.$imageNameNew;
	move_uploaded_file($imageTmp, $imageDest);
	
	exec("python3 ".$scriptFile." ".$imageDest." ".$tecnica." > result.txt 2>&1 &", $output);

	sleep(3);

	$resultFile = fopen("result.txt", "r");
	$result = fread($resultFile, filesize("result.txt"));
	fclose($resultFile);
	header("Location: resultado.php?image=".$imageDest."&tecnica=".$tecnica."&result=".$result);
}
?>