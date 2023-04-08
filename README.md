# psd-export

- Fast multi-threaded exporting of PSDs with `[tagged]` layers for variants.
- Special named layers can be used to apply filters like mosaic or gaussian blur, or whatever else you can hook in.
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
  - A layer name can contain anything else outside of tags: `scene [1]`, which will be ignored.
- Because whitespace is used to delimit arguments to filters, a tag's name will not include anything after a space, for example:
  - `[blur 50]` the tag name will be `blur`, and `50` is an argument to `blur`.
- Each set of primary tagged layers will be turned on and off and then exported in turn.
  - That means multiple layers can have the same tag, so you can toggle layers in your foreground and background together, for example:
    - A layer named `foreground [1]` and a separate layer named `background [1]`
- A layer can have multiple tags in the name: `scene [1][2]`
  - This will export with this layer visible for both tags.
- A primary tag is not necessary. If no primary tag is provided, then the whole picture is exported as is.

##### Secondary tags
- Tags with an `@` in the name will be treated as secondary tags.
- Text before the `@` is the tag name, and text after the `@` is the exclusion group.
  - The exclusion group can be empty.
  - Valid secondary tag names: `[jp@]`, `[jp@text]`
- These tags will be exported in combination with primary tags, for example:
  - If you have layers tagged `[1]`, `[thing@]` `[jp@text]`, and `[en@text]`, then what will be exported is `1`, `1-thing`, `1-thing-jp`, `1-thing-en`, `1-jp`, `1-en`
  - Because `[jp@text]` and `[en@text]` share the same exclusion group `text`, they will never be enabled together.
  - If there was a second primary tag `[2]` for example, the whole set of combinations would be exported again but with `2` instead of `1`

#### Image filters
- Tags with special names will be treated as filters on the colors below them.
- Filters can double as export tags too.
- If you want a filter to not be treated as an export tag, you can preceed it with `#` to set the ignore flag, for example `[#censor]`
- Filters can have arguments as well to control their behavior, separated by spaces, for example `[#censor 50]`
- If you want the filter to apply to layers outside of the group it is in, then the group blend mode should be set to `pass-through`, otherwise it may blend with transparent black pixels if the filter is over a transparent part of a group.
- If multiple filters are enabled in one layer, they will be applied from left to right on top of each result, example:
  - `[#censor][#blur]` will apply a mosaic, and then a blur on top of that mosaic.

##### Available default filters:
- `[censor mosaic_factor]`
  - If the mosaic_factor argument is omitted, then it is defaulted to 100, which means 1/100th the smallest dimension, (or 4 pixels, whichever is larger) in order to be Pixiv compliant.
  - Typically you will want this filter to be a secondary tag, for example: `[censor@]`, so you can have censored and uncensored outputs.
- `[blur size]`
  - The `size` argument defaults to 50 if omitted.
  - This filter is best used to create a non-destructive blur, such as for a background layer. You can fill an entire layer and set it to `[#blur 8]` for example.

##### Adding a new filter:
A filter function should take the form of:
```py
# Only positional arguments work right now.
def some_filter(color, alpha, arg1=default1, arg2=default2, ..., *_):
    # Cast arguments to your desired types, as they will come in as strings.
    arg1 = float(arg1)
    # Manipulate color and alpha numpy arrays, in-place if you want.
    # Always return the same shaped arrays as a tuple:
    return color, alpha
```

Add it to the filter lookup dictionary:
```py
export.filter_names['my-filter'] = some_filter
```

Apply it to a layer, for example:
`[my-filter 20.4]` or `[#my-filter]`, etc.

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
### Blendmode status:
| Blendmode | Issue |
| - | - |
| Normal | Pass |
| Multiply | Pass |
| Screen | Pass |
| Overlay | Broken, depends on Hard Light |
| Linear Burn (Shade) | Pass |
| Linear Dodge (Shine) | Pass |
| Linear Light (Shade/Shine) | Pass |
| Color Burn (Burn) | Broken when source alpha < 1 |
| Color Dodge (Dodge) | Broken when source alpha < 1 |
| Vivid Light (Burn/Dodge) | Broken |
| Soft Light | Pass |
| Hard Light | Broken, Multiply part is off when backdrop alpha < 1 |
| Pin Light | Broken |
| Hard Mix | Small precision error for near-black colors |
| Darken | Pass |
| Lighten | Pass |
| Darken Color | Pass |
| Lighten Color | Pass |
| Difference | Currently [TS] Difference, don't know how to implement yet |
| Exclude | Pass |
| Subtract | Pass |
| Divide | Pass |
| Hue | Broken, bad HSV computation? |
| Saturation | Broken, bad HSV computation? |
| Color | Pass |
| Luminosity | Pass |
| [TS] Linear Burn (Shade) | Pass |
| [TS] Linear Dodge (Shine) | Pass |
| [TS] Linear Light (Shade/Shine) | Pass |
| [TS] Color Burn (Burn) | Pass |
| [TS] Color Dodge (Dodge) | Pass |
| [TS] Vivid Light (Burn/Dodge) | Pass |
| [TS] Hard Mix | Small precision error for near-black colors |
| [TS] Difference | Pass |

### Missing PSD features:
- Other things that are not implemented (also not implemented in `psd-tools`):
  - Adjustment layers (gradient map, color balance, etc.)
  - Layer effects (shadow, glow, overlay, strokes, etc.)
  - Font rendering
  - Probably some other things I'm unaware of.

---
### TODO:
- Fix blend modes that don't quite work properly. This is low priority because I hardly use these modes myself or merge the results when painting.
- Binary package export
