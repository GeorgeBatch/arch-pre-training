### Citation

```
@inproceedings{gamper2020multiple,
  title={Multiple Instance Captioning: Learning Representations from 
Histopathology Textbooks and Articles},
  author={Gamper, Jevgenij and Rajpoot, Nasir},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```

## Dataset Usage Rules

1. The dataset provided here is for research purposes only. Commercial uses are not allowed. The data is licensed under the following license

   [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/)

   [![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

2. If you intend to publish research work that uses this dataset, you **must** cite our paper (as mentioned above), wherein the same dataset was first used.

## Columns

* `figure_id` - corresponds to the id of the bag
* `letter` - corresponds to the id of the instance within the bag. `single` indicates that there is only a single instance within a bag.
* `caption` - is the textual caption for that bag.
* `uuid` - is the unique image identifier of that instance.

