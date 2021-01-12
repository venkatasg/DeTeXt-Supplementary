# Generating the training set

# Generating symbol set of LaTeX symbols

Run `drawing.ipynb` ensuring that you have the uncompressed Detexifier training
data as a CSV. It's a little fiddly to get the CSV, but I converted the SQL
database to CSV using SQLPro Studio (who offer a great 1 year free trial for
students as of me writing this).

*Note if you're running this script on a Mac*: The script draws images and saves
them in folders that are named using the corresponding command id in the CSV.
However, there are certain command IDs that differ only in case (for example
`latex2e-OT1-_pi` and `latex2e-OT1-_Pi`) and Macs are (and have always been)
case insensitive by default. I recommend using the `css_class` field from the
`symbols.json` file in the Detexify training data folder. This also takes care
of some pesky filenames like `latex2e-OT1-_/`.

# Generating the SVGs for use as symbol assets in XCode

The steps to generate SVGs for usage in the macOS app are convoluted since I hacked 
together a solution that got me what I wanted in a few days. Broadly, this is what 
I did:

1. Generate `svg` files of all the LaTeX symbols you want using the `dvitosvg.sh` 
script and the `tech.tex` file.
2. Create [Custom SF Symbols][apple-sf] for each of the SVGs - you need to 
copy and paste the guidelines(`<g id="Guides">`), and wrap the paths for that symbol 
in  `<g id='Regular-M'>`.
3. If you open any of the SVG files in a vector drawing app, you will notice that 
the symbol is not aligned with the `Regular-M` guidelines, nor is it scaled. I fixed 
this by writing a custom [Sketch][sketch] plugin `sketch-plugin.js` and running 
it repeatedly using another custom [Keyboard Maestro][km] shortcut 
`Sketch run.kmacros`. I've included both the script and the shortcut in the folder, 
you need to load them in the respective apps and modify them as necessary for your 
use case (like changing the paths).

I'm sorry the documentation for this process isn't more exhaustive or complete -- 
my goal was to get it done, not to get it done in a organized, clean, reproducible 
way. If you have any questions, feel free to [email me][email]


<!-- Links -->
[apple-sf]: https://developer.apple.com/documentation/xcode/creating_custom_symbol_images_for_your_app