..
    MIT License

    Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

String handling
~~~~~~~~~~~~~~~

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed like
``Series.str.<function/property>``.

.. currentmodule:: hipdf
.. autosummary::
   :toctree: api/

   Series.str

.. currentmodule:: hipdf.core.column.string.StringMethods
.. autosummary::
   :toctree: api/

   byte_count
   capitalize
   cat
   center
   character_ngrams
   character_tokenize
   code_points
   contains
   count
   detokenize
   edit_distance
   edit_distance_matrix
   endswith
   extract
   filter_alphanum
   filter_characters
   filter_tokens
   find
   findall
   find_multiple
   get
   get_json_object
   hex_to_int
   htoi
   index
   insert
   ip2int
   ip_to_int
   is_consonant
   is_vowel
   isalnum
   isalpha
   isdecimal
   isdigit
   isempty
   isfloat
   ishex
   isinteger
   isipv4
   isspace
   islower
   isnumeric
   isupper
   istimestamp
   istitle
   jaccard_index
   join
   len
   like
   ljust
   lower
   lstrip
   match
   minhash
   ngrams
   ngrams_tokenize
   normalize_characters
   normalize_spaces
   pad
   partition
   porter_stemmer_measure
   repeat
   removeprefix
   removesuffix
   replace
   replace_tokens
   replace_with_backrefs
   rfind
   rindex
   rjust
   rpartition
   rsplit
   rstrip
   slice
   slice_from
   slice_replace
   split
   startswith
   strip
   swapcase
   title
   token_count
   tokenize
   translate
   upper
   url_decode
   url_encode
   wrap
   zfill
