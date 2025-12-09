---
license: apache-2.0
base_model:
- PekingU/rtdetr_v2_r18vd
pipeline_tag: object-detection
library_name: transformers.js
---

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library
from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Perform object-detection with `onnx-community/rtdetr_v2_r18vd-ONNX`.

```js
import { pipeline } from '@huggingface/transformers';

const detector = await pipeline('object-detection', 'onnx-community/rtdetr_v2_r18vd-ONNX');

const img = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
const output = await detector(img, { threshold: 0.75 });
console.log(output);
```
