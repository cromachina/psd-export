# psd-export
- Fast multi-threaded exporting of PSDs with [tagged] layers for variants.
- Special named layers can be used as masks to control special effects, like mosaic or gaussian blur.
- This tool is primarily meant to be compatible with PSDs exported from `PaintTool SAIv2` and `Clip Studio Paint`.

---
### Why
For my art workflow, I typically make a bunch variation layers and also need to apply mosaics to all of them. As a manual process, exporting every variant can take several minutes, with lots of clicking everywhere to turn layers on and off (potentially missing layers by accident) and then adding mosaics can be another 10 or 20 minutes of extra work. If I find I need to change something in my pictures and export again, I have to repeat this whole ordeal. This script puts this export process on the order of seconds, saving me a lot of time and pain.

---
### Installation
- Install python: https://www.python.org/downloads/
- Run `pip install -r requirements.txt` to install dependencies.

---
### Setting up the PSD

#### Automatic Exporting
##### Primary tags
- Add tags surrounded by `[]` to the names of layers that you want as part of an exported picture.
  - A tag name cannot be empty, so `[]` will not be recognized as a tag.
  - Valid tag names: `[1]`, `[hello]`
  - A layer name can contain anything else outside of tags: `scene [1]`
- Each set of primary tagged layers will be turned on and off and then exported in turn.
  - That means multiple layers can have the same tag, so you can toggle layers in your foreground and background together, for example.
- A layer can have multiple tags in the name: `scene [1][2]`
  - This will export with this layer visible for both tags.
- A primary tag is not necessary. If no primary tag is provided, then the whole picture is exported as is.
##### Secondary tags
- Tags with an `@` in the name will be treated as secondary tags.
- Text before the `@` is the tag name, and text after the `@` is the exclusion group.
  - The exclusion group can be empty.
  - Valid secondary tag names: `[jp@]`, `[jp@text]`
- These tags will be exported in combination with primary tags.
- Tags with the same exclusion group will be exported exclusively from each other.
  - Example: `[jp@text]`, `[en@text]`, these two layers will not be turned on together because they are in the exclusion group `text`.

#### Automatic Censors
- Any tag with the name `censor` will be turned into a mosaic filter within the layer group it resides in.
  - If you want the mosiac to apply to layers outside of the group it is in, then the group blend mode should be set to `pass-through`. If the mosaic blends over transparent pixels, then some of the mosaic pixels may blend with the black transparency if the group blend mode is not `pass-through`.
  - By default, the mosaic is Pixiv compliant, that is: 1/100th the smallest dimension, or 4 pixels, whichever is larger
- Typically you will want this to be a secondary tag, for example: `[censor@]`

---
### Running
- Run `export.py --file-name some-picture.psd` to run the export process.
- You can omit the file name and it will default to `*.psd` and try to export all PSD files in the current directory.
- Run `export.py --h` to see a list of arguments and their behaviors.
- Command argument names can be partially entered, for example `--subfolders` can be written `--sub` or `--s`, if it's unambiguous.

---
### Examples
In the layers below, there are 4 primary tags and 4 secondary tags (with two unique exclusion groups):

Primary tags: `1`, `2`, `3`, `4`

Secondary tags: `jp`, `en`, `hearts`, `censor`

Exclusion groups: `text`, `<empty>`

Groups with primary tags will be exported again even if the secondary tag does not exist under that group! This keeps the exported folders uniformly sized for publishing, so different folders may appear to have duplicate outputs.

Example layer configuration in SAI:

![image](https://user-images.githubusercontent.com/82557197/194391407-cc3dc945-24f9-4a28-be0c-6e579e432317.png)

Example output from script, showing every valid combination:

![image](https://user-images.githubusercontent.com/82557197/194395531-133e1650-d81f-4d27-9e23-36b2171e9ea2.png)

Folder after exporting everything:

![image](https://user-images.githubusercontent.com/82557197/194395730-d0c4a7b4-e332-4050-9a8b-2dec8cf1c9ce.png)

Example of a layer with the tag `[censor@]` (the layer does not need to be set to visible before exporting):

<img src="https://user-images.githubusercontent.com/82557197/194396108-6a1fa5f5-b311-43b2-959f-a999f69655af.png" width="500">

After exporting:

<img src="https://user-images.githubusercontent.com/82557197/194396262-0c2c3879-fa65-40a3-bc87-f0ddd779dd26.png" width="500">

---
### Issues:
- Some blend modes give a different result from SAI when the source layer has transparency, otherwise the result is the same. Typically this is because blend mode has some non-linear formula that I have trouble reverse engineering. These blend mode behaviors somewhat correspond to the 'Fill' parameter in Photoshop. Currently the behavior of these functions is defaulted to the Transparency Shapes [TS] blend modes in SAI, if there is a matching one.
  - Color Burn (Burn)
  - Color Dodge (Dodge)
  - Vivid Light (Burn/Dodge)
  - Difference

- Blend modes that mostly work, but look slightly different when the layer below has transparency:
  - Linear Dodge (slightly transparent things end up too bright)
  - Hard Mix (blending appears to be non-linear)

- Blend modes that mostly work, but have very different results when the layer below has transparency (0 alpha in a group backdrop will show a darker color than it should, could be an issue with black backdrops):
  - Linear Light
  - Hard Light
  - Pin Light

- Blend modes that have a slightly different result even at full opacity:
  - Soft Light

- Blend modes that seem to be completely broken:
  - Hue
  - Saturation

- Other things that are not implemented (also not implemented in `psd-tools`):
  - Adjustment layers (gradient map, color balance, etc.)
  - Layer effects (shadow, glow, overlay, strokes, etc.)
  - Font rendering
  - Probably some other things I'm unaware of.

---
### TODO:
- Fix blend modes that don't quite work properly. This is low priority because I hardly use these modes myself or merge the results when painting.
- Binary package export
