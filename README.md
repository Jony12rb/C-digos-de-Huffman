# Código de Huffman

## Descripción del proyecto
Este proyecto implementa las clases necesarias para poder registrar códigos fuente (sin memoria) y calcular, dado un alfabeto, un código de Huffman adiente a dicha fuente.

## Funcionamiento
Para crear un código de Huffman, bastará con crear un objeto de la clase `HuffmanCode`, pasándole como parámetros un `np.ndarray` con los símbolos de la fuente y otro con las probabilidades de cada símbolo. Como ejemplo, se puede ver el siguiente código:

```python
import huffman as hm
import numpy as np
code = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j" ])
prob = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)/55
fc = hm.FontCode(code, prob) 
```

Luego, podemos pedirle que nos calcule un código de Huffman para dicha fuente dado el alfabeto que queramos. El resultado será guardado en el
atributo `huffman_code` del objeto `fc` como un diccionario:

```python
q = [0, 1, 2, 3]
fc.huffman_enc(q) # Compute the Huffman code
print(fc.huffman_code)
```

La salida será:

```python
{'i': [0], 'j': [1], 'a': [2, 0], 'b': [2, 1], 'c': [2, 2], 'd': [2, 3], 'e': [3, 0], 'f': [3, 1], 'g': [3, 2], 'h': [3, 3]}
```
