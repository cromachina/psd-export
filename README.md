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
- Run `pip install psd-export` to install the script.
- Run `psd-export` to export any PSD files in the current directory.
- Run `psd-export --help` for more command line arguments.

---
### Building from source
- Install a C compiler (like MSVC, GCC, Clang)
- Install extra dependencies: `pip install setuptools wheel cython numpy`
- Install this repository locally: `pip install -e .`
- If you modify a Cython file (.pyx), then rebuild it with: `python setup.py build_ext --inplace`

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
- If you want the filter to apply to layers outside of the group it is in, then the group blend mode should be set to `pass-through`, otherwise it may blend with transparent black pixels if the filter is over a transparent part of a group. The blur and motion blur filters apply to alpha as well, so they should behave as expected in isolated groups.
- If multiple filters are enabled in one layer, they will be applied from left to right on top of each result, example:
  - `[#censor][#blur]` will apply a mosaic, and then a blur on top of that mosaic.
- Blend modes and clipping layers applied directly to filter layers are ignored.

##### Available default filters:
- `[censor mosaic_factor apply_to_alpha]`
  - If the `mosaic_factor` argument is omitted, then it is defaulted to 100, which means 1/100th the smallest dimension, (or 4 pixels, whichever is larger) in order to be Pixiv compliant.
  - `apply_to_alpha` defaults to False. Any value is treated as True.
  - Typically you will want this filter to be a secondary tag, for example: `[censor@]`, so you can have censored and uncensored outputs.
- `[blur size]`
  - The `size` argument defaults to 50 if omitted.
  - This filter is best used to create a non-destructive blur, such as for a background layer. You can fill an entire layer and set it to `[#blur 8]` for example.
- `[motion-blur angle size]`
  - `angle` is in degrees, starting from horizontal to the right; Default 0.
  - `size` defaults to 50.
  - Best used for non-destructive blur: `[#motion-blur 45 20]`

##### Adding a new filter:
In your own script:
```py
# my-export.py
from psd_export import (export, filters, util, blendfuncs)
import numpy as np

my_arg1_default = 1.0

# Register the filter with this decorator:
@filters.filter('my-filter')
# Only positional arguments work right now. The result of this function replaces the destination color and alpha.
def some_filter(color_dst, color_src, alpha_dst, alpha_src, arg1=None, arg2=100, *_):
    # Cast arguments to your desired types, as they will come in as strings.
    if arg1 is None:
        arg1 = my_arg1_default
    arg1 = float(arg1)
    # Manipulate color and alpha numpy arrays, in-place if you want.
    color = np.subtract(arg1, color, out=color)
    color = blendfuncs.lerp(color_dst, color, alpha_src)
    # Always return the same shaped arrays as a tuple:
    return color, alpha

if __name__ == '__main__':
    # Add your own command line arguments if needed.
    export.arg_parser.add_argument('--my-arg1', default=my_arg1_default, type=float,
        help='Set the arg1 default parameter.')
    args = export.arg_parser.parse_args()
    my_arg1_default = args.my_arg1

    export.main()
```

Apply it to a layer in your PSD, for example:
`[my-filter 20.4]` or `[#my-filter]`, etc.

---
### Examples
In the layers below, there are 2 primary tags and 3 secondary tags (with two unique exclusion groups):

Primary tags: `1`, `2`

Secondary tags: `jp`, `en`, `censor`

Exclusion groups: `text`, `<empty>`

Groups with primary tags will be exported again even if the secondary tag does not exist under that group! This keeps the exported folders uniformly sized for publishing, so different folders may appear to have duplicate outputs.

Example layer configuration in SAI:

![image](https://user-images.githubusercontent.com/82557197/232172462-a52cf239-0adc-4ad0-9601-8d6d79fd158d.png)

Example output from script, showing every valid combination:

![image](https://user-images.githubusercontent.com/82557197/232172483-20d1089f-6ccf-46f4-96ed-4651d9e7b1e7.png)

Folder after exporting everything:

![image](https://user-images.githubusercontent.com/82557197/232172514-39985f21-b5ac-4f63-9be6-483355dada4b.png)

Example of a layer with the tag `[censor@]` (the layer does not need to be set to visible before exporting):

![image](https://user-images.githubusercontent.com/82557197/232172599-7146521c-f539-4753-81ec-f21d4eb98aa9.png)

After exporting:

![image](https://user-images.githubusercontent.com/82557197/232172637-4b4e397c-53cb-4449-8525-bca0603d9ec1.png)

---
### Blendmode status:
| Blendmode | Status |
| - | - |
| Normal | Pass |
| Multiply | Pass |
| Screen | Pass |
| Overlay | Pass |
| Linear Burn (Shade) | Pass |
| Linear Dodge (Shine) | Pass |
| Linear Light (Shade/Shine) | Pass |
| Color Burn (Burn) | Pass |
| Color Dodge (Dodge) | Pass |
| Vivid Light (Burn/Dodge) | Pass |
| Soft Light | Pass |
| Hard Light | Pass |
| Pin Light | Pass |
| Hard Mix | Small precision error for near-black colors, can look slightly different from SAI |
| Darken | Pass |
| Lighten | Pass |
| Darken Color | Pass |
| Lighten Color | Pass |
| Difference | Pass |
| Exclude | Pass |
| Subtract | Pass |
| Divide | Pass |
| Hue | Pass |
| Saturation | Pass |
| Color | Pass |
| Luminosity | Pass |
| [TS] Linear Burn (Shade) | Pass |
| [TS] Linear Dodge (Shine) | Pass |
| [TS] Linear Light (Shade/Shine) | Pass |
| [TS] Color Burn (Burn) | Small precision error for near-black colors, can look slightly different from SAI |
| [TS] Color Dodge (Dodge) | Pass |
| [TS] Vivid Light (Burn/Dodge) | Pass |
| [TS] Hard Mix | Small precision error for near-black colors, can look slightly different from SAI |
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
