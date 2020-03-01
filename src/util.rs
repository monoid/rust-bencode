use std::{str, fmt};
use std::borrow::Borrow;

use serde::{self, Serialize};

// TODO: use bstr
#[derive(Eq, PartialEq, Clone, Ord, PartialOrd, Hash, Debug)]
pub struct ByteString(Vec<u8>);

impl ByteString {
    pub fn from_str(s: &str) -> ByteString {
        ByteString(s.as_bytes().to_vec())
    }

    pub fn from_slice(s: &[u8]) -> ByteString {
        ByteString(s.to_vec())
    }

    pub fn from_vec(s: Vec<u8>) -> ByteString {
        ByteString(s)
    }

    pub fn as_slice(&self) -> &[u8] {
      match self {
        &ByteString(ref v)  => &v[..]
      }
    }

    pub fn unwrap(self) -> Vec<u8> {
        let ByteString(v) = self;
        v
    }
}

impl fmt::Display for ByteString {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use fmt_bytestring;
        match self {
            &ByteString(ref v) => fmt_bytestring(&v[..], fmt),
        }
    }
}

impl Serialize for ByteString {
    fn serialize<S: serde::Serializer>(&self, e: S) -> Result<S::Ok, S::Error> {
        let &ByteString(ref key) = self;
        e.serialize_str(unsafe { str::from_utf8_unchecked(&key[..]) })
    }
}

impl Borrow<[u8]> for ByteString {
    fn borrow(&self) -> &[u8] {
        match self {
            &ByteString(ref v)  => v.borrow()
        }
    }
}

// FIXME: fix the error
//impl<'a> Decodable for ByteString {
    //fn decode(d: &mut super::Decoder<'a>) -> Result<ByteString, DecoderError> {
        //d.read_str().map(|s| ByteString(s.into_bytes())).or_else(|e| {
            //match e {
                //StringEncoding(v) => Ok(ByteString(v)),
                //err => Err(err)
            //}
        //})
    //}
//}

#[cfg(test)]
mod test {
    use super::ByteString;
    use super::super::encode;

    #[test]
    fn test_encoding_bytestring_with_invalid_utf8() {
        let bs = ByteString::from_slice(&[45, 18, 128]);
        let encoded = match encode(&bs) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        assert_eq!(vec![51, 58, 45, 18, 128], encoded);
    }

    // FIXME: fix decodable implementation
    //#[test]
    //fn test_decoding_bytestring_with_invalid_utf8() {
        //let encoded = vec![51, 58, 45, 18, 128];
        //let bencode = from_vec(encoded).unwrap();
        //let mut decoder = Decoder::new(&bencode);
        //let result: Result<ByteString, _> = Decodable::decode(&mut decoder);
        //assert_eq!(Ok(ByteString::from_slice(&[45, 18, 128])), result);
    //}
}
