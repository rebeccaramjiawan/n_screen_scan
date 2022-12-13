"""
This module contains a JSON encoder and decoder to save and load
`JSON <https://docs.python.org/3/library/json.html>`_ data in
a human-readable format.

So far, handling of the following objects have been re-implemented. Contributions to extend it are welcome.

* timestamps (``datetime.datetime``, ``datetime.date``, ``pytz.BaseTzInfo`` and ``datetime.timedelta``)
* numpy arrays (``numpy.ndarray``)

only a few datetime and numpy objects have been implemented.

Examples
--------

How to write (encode) a python dict ``mydata`` to a file ``mydata.json``:

.. code-block:: python

   import datetime
   import json
   import numpy
   from pybt.myjson.encoder import myJSONEncoder

   mydata = {'my_timestamp': datetime.datetime(2020, 12, 16, 21, 43, 43, 827438),
             'my_ndarray': numpy.linspace(0, 10)}

   with open('mydata.json', 'w') as outfile:
       outfile.write(json.dumps(mydata, cls=myJSONEncoder))


How to read (decode) data from a file ``mydata.json`` to python dict ``mydata``:

.. code-block:: python

   import json
   from pybt.myjson.encoder import myJSONDecoder

   with open('mydata.json') as infile:
       mydata = json.loads(infile.read(), cls=myJSONDecoder)

And to do it with compressed files one can us the gzip core python package. Note that compressed files are indeed no longer readable by baseline humans ...

.. code-block:: python

   import gzip

   with gzip.open('mydata.json.gz', 'wb') as outfile:
      outfile.write(gzip.compress(json.dumps(mydata, cls=myJSONEncoder).encode('utf-8')))

   with gzip.open('mydata.json.gz', 'rb') as infile:
      mydata = json.loads(gzip.decompress(infile.read()).decode('utf-8'), cls=myJSONDecoder)



"""

import json
import datetime
import dateutil.parser
import pytz
import numpy as np


class myJSONEncoder(json.JSONEncoder):
    """
    This subclass extends `json.JSONEncoder <https://docs.python.org/3/library/json.html#encoders-and-decoders>`_ to recognize other
    objects by implementing a ``default()`` method that returns a
    serializable object an encoder be used to save data from japc


    """

    def default(self, obj):
        # Datetime objects
        if isinstance(obj, datetime.datetime):
            return {
                '__type__': 'datetime',
                'data': obj.isoformat()
            }
        if isinstance(obj, datetime.date):
            return {
                '__type__': 'date',
                'data': obj.isoformat()
            }
        if isinstance(obj, pytz.BaseTzInfo):
            return {
                '__type__': 'pytz.BaseZzInfo',
                'data': str(obj)
            }
        if isinstance(obj, datetime.timedelta):
            return {
                '__type__': 'datetime.timedelta',
                'days': obj.days,
                'seconds': obj.seconds,
                'microseconds': obj.microseconds
            }
        # Numpy arrays
        if isinstance(obj, (np.ndarray,)):
            return {
                '__type__': 'np.ndarray',
                'data': obj.tolist()
            }
        return super(myJSONEncoder, self).default(obj)


class myJSONDecoder(json.JSONDecoder):
    """
    This subclass extends `json.JSONDencoder <https://docs.python.org/3/library/json.html#encoders-and-decoders>`_ to recognize default objects
    through hooks defined by ``__type__`` to deserialize the object in JSON
    and converted to python

    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('object_hook', self.default_object_hook)
        super(myJSONDecoder, self).__init__(*args, **kwargs)

    def default_object_hook(self, obj):
        if '__type__' in obj:
            if obj['__type__'] == 'datetime':
                return dateutil.parser.parse(obj['data'])
            if obj['__type__'] == 'date':
                return datetime.parser.parse(obj['data']).date()
            if obj['__type__'] == 'pytz.BaseZzInfo':
                return pytz.timezone(obj['data'])
            if obj['__type__'] == 'datetime.timedelta':
                return datetime.timedelta(days=obj['days'], seconds=obj['seconds'], microseconds=obj['microseconds'])
            if obj['__type__'] == 'np.ndarray':
                return np.asarray(obj['data'])
        return obj


