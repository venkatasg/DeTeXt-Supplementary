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

I will update this soon with the steps.