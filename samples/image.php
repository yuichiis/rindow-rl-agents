<?php
/*
$deg = 0;
$scalex = -1;
$scaley = 1;
$theta = $deg/180*M_PI;


function decompositionTest($scalex,$scaley,$theta)
{
    $xbase = [1,0];
    $ybase = [0,1];
    $xbase_r[0] = $scalex*cos($theta);
    $xbase_r[1] = $scaley*sin($theta);
    $ybase_r[0] = $scalex*cos($theta+M_PI/2);
    $ybase_r[1] = $scaley*sin($theta+M_PI/2);
    
    //echo "xbase_r=[".implode(',',$xbase_r)."],ybase_r=[".implode(',',$ybase_r)."]\n";
    $dxscale=sqrt($xbase_r[0]**2+$xbase_r[1]**2);
    $dyscale=sqrt($ybase_r[0]**2+$ybase_r[1]**2);
    if($xbase_r[0]==0) {
        $xth = NAN;
    } else {
        $xth = atan($xbase_r[1]/$xbase_r[0]);
    }
    if($xbase_r[1]==0) {
        $yth = NAN;
    } else {
        $yth = atan($xbase_r[0]/$xbase_r[1]);
    }
    //echo "xth=$xth,yth=$yth\n";
    //return [null,$xbase_r,$ybase_r,[$dxscale,$dyscale],[],0,$xth,$yth];

    $illegal = false;
    if(abs($xbase_r[0])>abs($xbase_r[1])) {
        if($xbase_r[0]>=0&&$ybase_r[1]>=0) {
            $th = atan($xbase_r[1]/$xbase_r[0]);       // ==== Front side 1 ====
            $scale = [1,1];
            $flags = 'F1';
        } elseif($xbase_r[0]<0&&$ybase_r[1]<0) {
            $th = atan($xbase_r[1]/$xbase_r[0])+M_PI;  // ==== Front side 2 ====
            $scale = [1,1];
            $flags = 'F2';
        } elseif($xbase_r[0]>=0&&$ybase_r[1]<0) {
            $th = -atan($xbase_r[1]/$xbase_r[0])-M_PI; // ==== Back side 1 ====
            $scale = [-1,1];
            $flags = 'B1';
        } elseif($xbase_r[0]<0&&$ybase_r[1]>=0) {
            $th = -atan($xbase_r[1]/$xbase_r[0]);      // ==== Back side 2 ====
            $scale = [-1,1];
            $flags = 'B2';
        } else {
            $illegal = true;
        }
    } else {
        if($xbase_r[1]>=0&&$ybase_r[0]<0) {
            $th = M_PI/2-atan($xbase_r[0]/$xbase_r[1]);   // ==== Front side 3 ====
            $scale = [1,1];
            $flags = 'F3';
        } elseif($xbase_r[1]<0&&$ybase_r[0]>=0) {
            $th = -M_PI/2-atan($xbase_r[0]/$xbase_r[1]);  // ==== Front side 4 ====
            $scale = [1,1];
            $flags = 'F4';
        } elseif($xbase_r[1]>=0&&$ybase_r[0]>=0) {
            $th = M_PI/2+atan($xbase_r[0]/$xbase_r[1]);   // ==== Back side 3 ====
            $scale = [-1,1];
            $flags = 'B3';
        } elseif($xbase_r[1]<0&&$ybase_r[0]<0) {
            $th = -M_PI/2+atan($xbase_r[0]/$xbase_r[1]);  // ==== Back side 4 ====
            $scale = [-1,1];
            $flags = 'B4';
        } else {
            $illegal = true;
        }
    }
    if($illegal) {
        echo "scalex=".$scalex.",scaley=".$scaley.",rotate=".sprintf("%2.0f",$theta*180/M_PI)."\n";
        echo "xb=".implode(',',$xbase_r)." yb=".implode(',',$ybase_r)."\n";
        throw new \Exception("Illegal");
    }
    if($th>M_PI) {
        $th -= 2*M_PI;
    }
    if($th<-M_PI) {
        $th += 2*M_PI;
    }
    return [$flags,$xbase_r,$ybase_r,[$dxscale,$dyscale],$scale,$th,$xth,$yth];
}


$data = [
    [1,1,-180],
    [1,1,-179],
    [1,1,-135],
    [1,1,-90],
    [1,1,-45],
    [1,1,-1],
    [1,1,0],
    [1,1,1],
    [1,1,45],
    [1,1,90],
    [1,1,135],
    [1,1,179],
    [1,1,180],

    [NAN,NAN,NAN],

    [-1,1,-180],
    [-1,1,-179],
    [-1,1,-135],
    [-1,1,-90],
    [-1,1,-45],
    [-1,1,-1],
    [-1,1,0],
    [-1,1,1],
    [-1,1,45],
    [-1,1,90],
    [-1,1,135],
    [-1,1,179],
    [-1,1,180],

    [NAN,NAN,NAN],

    [-1,-1,-180],
    [-1,-1,-135],
    [-1,-1,-90],
    [-1,-1,-45],
    [-1,-1,0],
    [-1,-1,45],
    [-1,-1,90],
    [-1,-1,135],
    [-1,-1,180],

    [NAN,NAN,NAN],

    [1,-1,-180],
    [1,-1,-135],
    [1,-1,-90],
    [1,-1,-45],
    [1,-1,0],
    [1,-1,45],
    [1,-1,90],
    [1,-1,135],
    [1,-1,180],
];

function signflags(array $a)
{
    $flags = '';
    $axisx = true;
    foreach ($a as $key => $x) {
        if($x==0) {
            if($axisx) {
                $flags .= 'Z';
            } else {
                $flags .= 'Z';
            }
        } else
        if($x>0) {
            $flags .= 'P';
        } else {
            $flags .= 'M';
        }
        $axisx = !$axisx;
    }
    return $flags;
}

$signtable = [];
$stage = 0;
$part = 0;
foreach($data as $d) {
    [$scalex,$scaley,$deg] = $d;
    if(is_nan($scalex)) {
        echo "==========\n";
        $stage++;
        $part = 0;
        continue;
    }
    $theta = $deg/180*M_PI;
    [$f,$xbase_r,$ybase_r,$dscale,$scale,$th,$xth,$yth] = decompositionTest($scalex,$scaley,$theta);
    $flags = signflags(array_merge($xbase_r,$ybase_r));
    if(!isset($signtable[$flags])) {
        $signtable[$flags] = $stage.$part;
        $part++;
    }

    if($scalex*$scaley>0) {
        $dtx = $deg-$xth*180/M_PI;
        $dty = $deg+$yth*180/M_PI;
    } else {
        $dtx = $deg+$xth*180/M_PI;
        $dty = $deg-$yth*180/M_PI;
    }
    echo sprintf("%s[%2.0f,%2.0f,%4.0f]=".
        "x=[%4.1f,%4.1f],y=[%4.1f,%4.1f],".
        "th=[%4.0f],".
        //"t=[%4.0f,%4.0f],".
        "s=[%4.1f,%4.1f]".
        //"d=[%4.0f,%4.0f]".
        "\n",
        $f,//($signtable[$flags].$flags),
        $scalex,$scaley,$deg,
        $xbase_r[0],$xbase_r[1],$ybase_r[0],$ybase_r[1],
        $th*180/M_PI,
        //$xth*180/M_PI,$yth*180/M_PI,
        $scale[0],$scale[1]
        //$dtx,$dty
    );
    //echo sprintf("%s[%2.0f,%2.0f,%4.0f]=".
    //    "x=[%4.1f,%4.1f],y=[%4.1f,%4.1f],".
    //    //"th=[%4.0f],".
    //    "t=[%4.0f,%4.0f],".
    //    //"s=[%4.1f,%4.1f]".
    //    "d=[%4.0f,%4.0f]".
    //    "\n",
    //    ($signtable[$flags].$flags),
    //    $scalex,$scaley,$deg,
    //    $xbase_r[0],$xbase_r[1],$ybase_r[0],$ybase_r[1],
    //    //$th*180/M_PI,
    //    $xth*180/M_PI,$yth*180/M_PI,
    //    //$dscale[0],$dscale[1]
    //    $dtx,$dty
    //);
}
//foreach ($signtable as $key => $value) {
//    echo $key."\n";
//}

exit();
*/
/*
if($x==0) {
    if($y>0) {
        $theta = 90/180*M_PI;
    } else {
        $theta = -90/180*M_PI;
    }
} else {
    $theta = atan($y/$x);
    if($x<0) {
        if($y>0) {
            $theta += M_PI;
        } else {
            $theta -= M_PI;
        }
    }
}
echo "deg=".($theta*180/M_PI)."\n";
echo "scale=".(sqrt($x**2+$y**2))."\n";
exit();
*/
/*
$dst = imagecreatetruecolor(100,100);
$yellow = imagecolorallocatealpha($dst,255,255,0,0);
imagefilledrectangle($dst,0,0,99,99,$yellow);

$src = imagecreatetruecolor(50,50);
$blue = imagecolorallocatealpha($src,0,0,255,0);
imagefilledrectangle($src,0,0,49,49,$blue);
$trans = imagecolorallocatealpha($src,0,0,0,127);

$rol = imagerotate($src,45,$trans);
$rolsx = imagesx($rol);
$rolsy = imagesy($rol);

imagecopy($dst,$rol,0,0,0,0,$rolsx-1,$rolsy-1);
$red = imagecolorallocatealpha($dst,255,0,0,0);
imagerectangle($dst,0,0,49,49,$red);
imagepolygon($dst,[0,99,99,99,49,0],3,$red);
imagepng($dst,'image.png');

imagedestroy($dst);
imagedestroy($src);
imagedestroy($rol);

system('image.png');
exit();
*/

$mo = new MatrixOperator();
$la = $mo->laRawMode();
$RAD2DEG = 57.29577951308232;
$fa = new RenderFactory($la,'gd');
$viewer = $fa->Viewer(500, 500);
$rendering = $viewer->rendering();
$viewer->set_bounds(-2.2, 2.2, -2.2, 2.2);

$fname = __DIR__.'/image.png';
$image = $rendering->Image($fname, 1.0, 1.0);
$imagetrans = $rendering->Transform();
$image->add_attr($imagetrans);
$viewer->add_geom($image);

$tri = $rendering->PolyLine([[-0.5,-0.5],[0.5,-0.5],[0.0,0.5]],true);
$boxtrans = $rendering->Transform();
$tri->add_attr($boxtrans);
$viewer->add_geom($tri);
$box = $rendering->PolyLine([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]],true);
$box->add_attr($boxtrans);
$viewer->add_geom($box);
$box2 = $rendering->PolyLine([[-0.5,0.0],[0.0,0.0],[0.0,0.5],[-0.5,0.5]],true);
$box2->add_attr($boxtrans);
$viewer->add_geom($box2);

$center = $rendering->PolyLine([[-0.01,-0.01],[0.01,-0.01],[0.01,0.01],[-0.01,0.01]],true);
$viewer->add_geom($center);

$imagetrans->set_translation(0,0);
$boxtrans->set_translation(0,0);
$imagetrans->set_scale(1,1);
$boxtrans->set_scale(1,1);
$imagetrans->set_rotation(0/$RAD2DEG);
$boxtrans->set_rotation(0/$RAD2DEG);
$viewer->render();
$viewer->show();

exit();
