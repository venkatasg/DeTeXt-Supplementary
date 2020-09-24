//cmd j
//console.log('This is the script to get svg in right position.')

var sketch = require('sketch')

var document = sketch.getSelectedDocument()

var left = document.getLayersNamed("left-margin")[0]

var right = document.getLayersNamed("right-margin")[0]

var bm = document.getLayersNamed("Baseline-M")[0]

var cm = document.getLayersNamed("Capline-M")[0]

var symb = document.getLayersNamed("Regular-M")[0]


//console.log(symb[0])
selection = document.selectedLayers
console.log(((left.frame.x + right.frame.x)/2 + left.frame.width/2) - symb.frame.width/2)
console.log(left.frame.y + (left.frame.height/2) - symb.frame.height/2)

// Set size of the symbol here
var maxWidth = right.frame.x - left.frame.x - left.frame.width/2
var maxHeight = bm.frame.y - cm.frame.y -cm.frame.height/2
var aspect = symb.frame.width/symb.frame.height

console.log(aspect, maxWidth, maxHeight)

if (aspect < 1) {
  symb.frame.height = maxHeight-1
  symb.frame.width = symb.frame.height*aspect
}

else {
  symb.frame.width = maxWidth-1
  symb.frame.height = symb.frame.width/aspect
}

// Now to position the symbol
symb.frame.x = 1360
symb.frame.y = 1010

// Export
//sketch.export(document.pages[0].layers[0], {formats: 'svg'})
