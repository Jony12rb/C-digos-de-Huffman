from __future__ import annotations # for Python 3.7 or later
import numpy as np
from typing import Generic, TypeVar, Union, Optional, List, Dict


T = TypeVar('T')

class HuffmanNode(Generic[T]):
    prob: float
    sons: Union[list[HuffmanNode], List[T]]
    
    
    def __init__(self, prob, sons):
        self.prob = prob
        self.sons = sons
        
    
    def __gt__(self, other):
        return self.prob > other.prob
    
    def __repr__(self) -> str:
        return f'HuffmanNode({self.prob}, {self.sons})'
        

    

class FontCode(Generic[T]):
    """
    A font code is a list of symbols and their probabilities. It can be
    encoded using the Huffman algorithm with base q. The code is
    a dictionary that maps each symbol to its code.
    """
    _code: np.ndarray[T]
    _prob: np.ndarray
    _hcode : Optional[Dict[T, List]] # Huffman code 
    
    
    def __init__(self, code, prob, *e):
        if(len(code) != len(prob)):
            raise ValueError("code and prob must have the same length")
        if(np.sum(prob) != 1): # check if prob is a probability distribution
            raise ValueError("prob must sum to 1")
        if e:
            raise TypeError("FontCode takes 2 positional arguments but {} were given".format(2+len(e)))
        self._code = code
        self._prob = prob
        self._hcode = {}
        
        
    @property
    def code(self):
        return self._code
    
    @property
    def prob(self):
        return self._prob
    
    @property
    def huffman_code(self):
        if self._hcode is None:
            raise ValueError("Huffman code not computed")
        return self._hcode
        
    
    def huffman_enc(self, q: np.ndarray):
        """
        Encode the code using the Huffman algorithm with base q
        """
        ql = len(q)
        if(ql < 2): # Encoding is not possible
            raise ValueError("q must have at least 2 elements")
        ccode = np.copy(self._code) 
        cprob = np.copy(self._prob)
        # Check if the size of the code is correct
        if len(ccode)%(ql-1) != 1 and ql != 2: 
            ccode = np.append(ccode,
                              np.full((ql-len(ccode)%(ql-1)), None));
            cprob = np.append(cprob,
                              np.full((ql-len(cprob)%(ql-1)), 0.0));
        self._enc(self._HuffTree(ccode, cprob, ql), q, [])
        
            
    def _HuffTree(self, code, prob, q):
        """
        Compute the Huffman tree of the code and return the root
        """
        # Initialize the list of nodes
        l = np.array([HuffmanNode(prob[i], code[i]) for i in range(len(code))])
        while len(l) > 1: # While there is more than one node
            l = np.sort(l) # Sort the nodes by probability
            l1 = l[:q]
            # Create a new node with the q first nodes as sons
            l = np.append(l[q:], HuffmanNode(np.sum([el.prob for el in l1]), l1)) 
        return l[0]
    
    
    def _enc(self, tree, q, code : List):
        """
        Given a Huffman tree, compute the Huffman code and store it in self._hcode
        """
        if type(tree.sons) is not np.ndarray: # If the node is a leaf
            if tree.sons is not None:
                self._hcode[tree.sons] = code              
        else:
            for i, el in enumerate(q):
                # Recursively compute the code for each son
                self._enc(tree.sons[i], q, code + [el]) 
                
                
    def expected_leght(self):
        """
        return the expected length of the code
        """
        if self._hcode is None:
            raise ValueError("Huffman code not computed")
        return np.sum([self._prob[i]*len(self._hcode[self._code[i]])
                       for i in range(len(self._prob))])
    
    
    def entropy(self, q): # q-base entropy
        """
        return the entropy of the code
        """
        if self._hcode is None:
            raise ValueError("Huffman code not computed")
        return np.sum(-1*self._prob*np.log2(self._prob)/np.log2(q))
        
    
    
def main():
    code = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j" ])
    prob = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)/55
    q = [0, 1, 2, 3]
    fc = FontCode(code, prob) # Create a FontCode object
    fc.huffman_enc(q) # Compute the Huffman code
    print(fc.huffman_code)
    print("Best possible lengt", fc.entropy(len(q)),
          "\nExpected length: ", fc.expected_leght())
 
    
if __name__ == "__main__": # execute only if run as a script
    main()