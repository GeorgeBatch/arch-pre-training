### Books Set

Some images from the `books_set` may contain letters on top. This letters are there to understand which part of the caption refers to which the image when multiple images correspond to one caption.

* Total Images (`./datasets/ARCH/books_set/images/`): 4270
* Total Caption Rows (`./datasets/ARCH/books_set/captions.json`): 4305
* **Captions with missing images**: 35

All of the images in the `./datasets/ARCH/books_set/images/` directory have a corresponding caption, but not all captions have a corresponding image.

**This table was computed from captions.json:**

Bag Size | # Bags
-------- | ------
1        | 2720   
2        | 378
3        | 133
4        | 56
5        | 12
6        | 14
7        | 4
8        | 2
9        | 2

`# Bags` is calculated using `figure_id` field, not `caption` field - not sure if it's the right way.

* Total unique captions: 3241
* Total unique figure_ids: 3321

**Due to the missing images, the values need to be recomputed.**

Bag Size | # Bags | Difference
-------- | ------ | ---------
1        | 2688   | 32
2        | 378    |
3        | 132    | 1
4        | 56     |
5        | 12     |
6        | 14     |
7        | 4      |
8        | 2      |
9        | 2      |

`# Bags` is calculated using `figure_id` field, not `caption` field - not sure if it's the right way.


Total difference is 35 = 32\*1 + 1\*3 images.

* Total unique captions: 3210
* Total unique figure ids: 3288

**Note: there is a difference of 78. Why can it be?**

* For each of the figure ids, there is always a single caption.

* However, the converse does not hold. There are 77 captions, which correspond to 2 (76 captions) or more (1 caption has 3 ids: ['4122', '4122', '4123', '4123', '4124']) different ids. In total, this gives a total difference between the number of unique captions and unique figure ids of **78**=76\*(2-1)+1\*(3-1). So the difference of **78=3288-3210** is explained by it.

TODO: understand if this is a mistake or it's ok. Emailed Jev Gamper (author).

**Calculating the number of bags using `caption`.**

**With missing images**

Bag Size | # Bags | Difference
-------- | ------ | ---------
1        | 2575   |
2        | 438    |
3        | 133    |
4        | 57     |
5        | 15     |
6        | 15     |
7        | 4      |
8        | 2      |
9        | 2      |

**Without missing images**

Bag Size | # Bags | Difference
-------- | ------ | ---------
1        | 2546   | 29
2        | 438    |
3        | 131    | 2
4        | 57     |
5        | 15     |
6        | 15     |
7        | 4      |
8        | 2      |
9        | 2      |

Total difference is 35 = 29\*1 + 2\*3 images (same as when counting using `figure_id`).

### PubMed Set

* Total Images (`./datasets/ARCH/pubmed_set/images/`): 3309 **(3272 jpg, 37 png)**
* Total Caption Rows (`./datasets/ARCH/pubmed_set/captions.json`): 3309
* Captions with missing images: 0

* Total Unique Captions: 3285
* Total Unique uuids: 3309
* 24 "extra captions"

Bag Size | # Bags
-------- | ------
1        | 3270
2        | 11
3        | 2
4        | 0
5        | 1
6        | 1

* Total unique captions: 3285 = 3270 + 11 + 2 + 0 + 1 + 1

**Captions are not split into different images. There are no "A", "B", "C" parts in a caption. There are also no "A", "B", "C" labels on images. This means that images with the same caption can be put in a bag with the caption, but also can probably be given to the model one by one.**

There are 15 = 11 + 2 + 0 + 1 + 1 captions with more than 1 uuid. In total, there are 24 = 11\*(2-1) + 2\*(3-1) + 0\*(4-1) + 1\*(5-1) + 1\*(6-1) = 11 + 4 + 0 + 4 + 5 extra captions.

**TODO: Ask Jev Gamper how they dealt with them. Did they put them in a batch?**

### Together (only counting when images are available)

Bag Size | # Bags
-------- | ------
1        | 5958   
2        | 389
3        | 134
4        | 56
5        | 13
6        | 15
7        | 4
8        | 2
9        | 2

* Total Images (`./datasets/ARCH/*/images/`): 7579 = 3309 + 4270
* Total Captions
