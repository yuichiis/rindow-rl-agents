<?php
namespace Rindow\RL\Agents\Util;

use RuntimeException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\NDArrayPhp;

class GDUtil
{
    public function __construct($la)
    {
        $this->la = $la;
    }

    public function getArray($gd)
    {
        if(!class_exists('Rindow\\Math\\Matrix\\NDArrayPhp')) {
            throw new LogicException('Requires rindow-math-matrix package.');
        }
        imageflip($gd,IMG_FLIP_VERTICAL);
        ob_start();
        imagebmp($gd);
        $bmp = ob_get_contents();
        ob_end_clean();
        imageflip($gd,IMG_FLIP_VERTICAL);
        $header1 = unpack("c2type/Vsize/v2rsv/Voffbits",$bmp,0);
        $header2 = unpack("Vbisize/Vwidth/Vheight/vplanes/vbitcount/".
                        "Vcomp/Vsizeimage/Vxpixpm/Vypixpm/".
                        "Vclrused/Vcirimp",$bmp,14);
        if($header2['comp']!=0) {
            throw new RuntimeException('bitmap format must be uncompressed.');
        }
        if($header2['bitcount']==24) {
            $channels = 3;
        } elseif($header2['bitcount']==32) {
            $channels = 4;
        } else {
            throw new RuntimeException('bitmap format must be 24 bit or 32 bit.');
        }
        $width = $header2['width'];
        $height = $header2['height'];
        $pxdata = '';
        $pos = $header1['offbits'];
        $linesize = (int)($width*$channels);
        $boundary = ($linesize&0x03)?(4-($linesize&0x03)):0;
        for($y=0;$y<$height;$y++) {
            $pxdata .= substr($bmp,$pos,$linesize);
            $pos += ($linesize+$boundary);
        }
        $la = $this->la;
        //$img = $la->alloc([$height,$width,$channels],NDArray::uint8);
        $img = new NDArrayPhp(null,NDArray::uint8,[$height,$width,$channels]);
        $buffer = $img->buffer();
        if(method_exists($buffer,'load')) {
            // OpenBlasBuffer
            $buffer->load($pxdata);
        } else {
            // SplFixedArray
            $idx=0;
            foreach(unpack('C*',$pxdata) as $value) {
                $buffer[$idx] = $value;
                $idx++;
            }
        }
        $img = $la->array($img);
        $img = $la->imagecopy($img,null,null,null,null,null,null,$rgbFlip=true);
        return $img;
    }
}
