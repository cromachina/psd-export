# psd-export
- Fast multi-threaded exporting of PSDs with [tagged] layers for variants.
- Special named layers can be used as masks to control special effects, like mosaic or gaussian blur.
- This tool is primarily meant to be compatible with PSDs exported from `PaintTool SAIv2` and `Clip Studio Paint`.

### Issues:
- Some blend modes give a different result from SAI when the source layer has a transparency, otherwise the result is the same. Typically this is because blend mode has some non-linear formula that I have trouble reverse engineering. These blend mode behaviors somewhat correspond to the 'Fill' parameter in Photoshop. Currently the behavior of these functions is defaulted to the Transparency Shapes [TS] blend modes in SAI, if there is a matching one.
  - Color Burn (Burn)
  - Color Dodge (Dodge)
  - Vivid Light (Burn/Dodge)
  - Hard Mix
  - Difference
  - Subtract (No matching TS mode)
  - Hue (No matching TS mode)

- Blend modes that have a slightly different result even at full opacity:
  - Soft Light

- Blend modes that seem to be completely broken:
  - Color

### TODO:
- Fix blend modes that don't quite work properly. This is low priority because I hardly use these modes myself or merge the results when painting.
- Add compatibility for CSP
- Install instructions
- Control instructions
- Mosaic composited under other layers:
  - Needs to be hooked into the compositor somehow, maybe a transformer function hook so anything can be used (like gaussian blur instead).
  - Kernel-like functions will need composite tile data padded to the kernel size. Tiles on the image boundary can be padded with a clamp
- Binary package export
