# psd-export
- Fast multi-threaded exporting of PSDs with [tagged] layers for variants.
- Special named layers can be used as masks to control special effects, like mosaic or gaussian blur.
- This tool is primarily meant to be compatible with PSDs exported from `PaintTool SAIv2` and `Clip Studio Paint`

### TODO:
- Finish blend modes for SAI and "8 special" compatiblity
- Install instructions
- Control instructions
- Mosaic composited under other layers:
  - Needs to be hooked into the compositor somehow, maybe a transformer function hook so anything can be used (like gaussian blur instead).
  - Kernel-like functions will need composite tile data padded to the kernel size. Tiles on the image boundary can be padded with a clamp
- Binary package export
