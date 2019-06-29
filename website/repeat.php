<?php

if(isset($_POST['submit'])){

    $image = $_POST['image'];
    $tecnica = $_POST['tecnica'];

    $scriptFile = "script.py";
	
	exec("python3 ".$scriptFile." ".$image." ".$tecnica." > result.txt 2>&1 &", $output);
	sleep(3);

	$resultFile = fopen("result.txt", "r");
	$result = fread($resultFile, filesize("result.txt"));
	fclose($resultFile);
	header("Location: resultado.php?image=".$image."&tecnica=".$tecnica."&result=".$result);
}

?>