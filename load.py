from core.datasets.fourchan.vmfunc_4chan_pol_extensive import VMFunc4ChanPolExtensive
from core.datasets.fourchan.sicariussicariistuff_ubw_tapestries import SicariusUBWTapestries
from core.datasets.safety.simplesafety import BertievidgenSimpleSafetyTests
import json

dataset = BertievidgenSimpleSafetyTests()

print(dataset.df)
