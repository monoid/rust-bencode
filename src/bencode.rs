// Copyright 2014 Arjan Topolovec
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "bencode"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]

/*!
  Bencode parsing and serialization

  # Encoding

  ## Using `Serialize`

  ```rust
  extern crate serde;
  extern crate bencode;

  use serde::Serialize;

  use bencode::encode;

  #[derive(Serialize)]
  struct MyStruct {
      string: String,
      id: usize,
  }

  fn main() {
      let s = MyStruct { string: "Hello bencode".to_string(), id: 1 };
      let result: Vec<u8> = encode(&s).unwrap();
  }
  ```

  ## Using `ToBencode`

  ```rust
  extern crate bencode;

  use std::collections::BTreeMap;

  use bencode::{Bencode, ToBencode};
  use bencode::util::ByteString;

  struct MyStruct {
      a: isize,
      b: String,
      c: Vec<u8>,
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = BTreeMap::new();
          m.insert(ByteString::from_str("a"), self.a.to_bencode());
          m.insert(ByteString::from_str("b"), self.b.to_bencode());
          m.insert(ByteString::from_str("c"), Bencode::ByteString(self.c.to_vec()));
          Bencode::Dict(m)
      }
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: vec![1, 2, 3, 4] };
      let bencode: bencode::Bencode = s.to_bencode();
      let result: Vec<u8> = bencode.to_bytes().unwrap();
  }

  ```

  # Decoding

  ## Using `Deserialize`

  ```rust
  extern crate serde;
  extern crate bencode;

  use serde::{Serialize, Deserialize};

  use bencode::{encode, Decoder};

  #[derive(Serialize, Deserialize, PartialEq)]
  struct MyStruct {
      a: i32,
      b: String,
      c: Vec<u8>,
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: vec![1, 2, 3, 4] };
      let enc: Vec<u8> = encode(&s).unwrap();

      let bencode: bencode::Bencode = bencode::from_vec(enc).unwrap();
      let mut decoder = Decoder::new(&bencode);
      let result: MyStruct = Deserialize::deserialize(&mut decoder).unwrap();
      assert!(s == result)
  }
  ```

  ## Using `FromBencode`

  ```rust
  extern crate bencode;

  use std::collections::BTreeMap;

  use bencode::{FromBencode, ToBencode, Bencode, NumFromBencodeError};
  use bencode::util::ByteString;

  #[derive(PartialEq)]
  struct MyStruct {
      a: i32
  }

  #[derive(Debug)]
  enum MyError {
      NotADict,
      DoesntContainA,
      ANotANumber(NumFromBencodeError),
  }

  impl ToBencode for MyStruct {
      fn to_bencode(&self) -> bencode::Bencode {
          let mut m = BTreeMap::new();
          m.insert(ByteString::from_str("a"), self.a.to_bencode());
          Bencode::Dict(m)
      }
  }

  impl FromBencode for MyStruct {
      type Err = MyError;

      fn from_bencode(bencode: &bencode::Bencode) -> Result<MyStruct, MyError> {
          use MyError::*;
          match bencode {
              &Bencode::Dict(ref m) => {
                  match m.get(&ByteString::from_str("a")) {
                      Some(a) => FromBencode::from_bencode(a).map(|a| {
                          MyStruct{ a: a }
                      }).map_err(ANotANumber),
                      _ => Err(DoesntContainA)
                  }
              }
              _ => Err(NotADict)
          }
      }
  }

  fn main() {
      let s = MyStruct{ a: 5 };
      let enc: Vec<u8>  = s.to_bencode().to_bytes().unwrap();

      let bencode: bencode::Bencode = bencode::from_vec(enc).unwrap();
      let result: MyStruct = FromBencode::from_bencode(&bencode).unwrap();
      assert!(s == result)
  }
  ```

  ## Using Streaming Parser

  ```rust
  extern crate serde;
  extern crate bencode;

  use bencode::streaming::BencodeEvent;
  use bencode::streaming::StreamingParser;
  use serde::{Serialize, Deserialize};

  use bencode::encode;

  #[derive(Serialize, Deserialize, PartialEq)]
  struct MyStruct {
      a: i32,
      b: String,
      c: Vec<u8>,
  }

  fn main() {
      let s = MyStruct{ a: 5, b: "foo".to_string(), c: vec![2, 2, 3, 4] };
      let enc: Vec<u8> = encode(&s).unwrap();

      let mut streaming = StreamingParser::new(enc.into_iter());
      for event in streaming {
          match event {
              BencodeEvent::DictStart => println!("dict start"),
              BencodeEvent::DictEnd => println!("dict end"),
              BencodeEvent::NumberValue(n) => println!("number = {}", n),
              // ...
              _ => println!("Unhandled event: {:?}", event)
          }
      }
  }
  ```
*/

#![cfg_attr(all(test, feature = "nightly"), feature(test))]

extern crate serde;
extern crate byteorder;
extern crate num_traits;

use std::io::{self, Cursor};
use std::fmt;
use std::str::{self, Utf8Error};
use std::vec::Vec;
use std::mem::size_of;

use serde::{de, ser, Serialize, forward_to_deserialize_any};

use byteorder::{ReadBytesExt, WriteBytesExt, BigEndian};
use num_traits::FromPrimitive;

use std::collections::BTreeMap;
use std::collections::HashMap;

use streaming::{StreamingParser, Error};
use streaming::BencodeEvent;
use streaming::BencodeEvent::{NumberValue, ByteStringValue, ListStart, ListEnd,
                              DictStart, DictKey, DictEnd, ParseError};
use self::Bencode::*;
use self::DecoderError::{Message, Unimplemented, Expecting, StringEncoding};

pub mod streaming;
pub mod util;

#[inline]
fn fmt_bytestring(s: &[u8], fmt: &mut fmt::Formatter) -> fmt::Result {
  match str::from_utf8(s) {
    Ok(s) => write!(fmt, "s\"{}\"", s),
    Err(..) => write!(fmt, "s{:?}", s),
  }
}

#[derive(PartialEq, Clone, Debug)]
pub enum Bencode {
    Empty,
    Number(i64),
    ByteString(Vec<u8>),
    List(ListVec),
    Dict(DictMap),
}

impl fmt::Display for Bencode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        format(fmt, self)
    }
}

fn format(fmt: &mut fmt::Formatter, v: &Bencode) -> fmt::Result {
    match *v {
        Bencode::Empty => { Ok(()) }
        Bencode::Number(v) => write!(fmt, "{}", v),
        Bencode::ByteString(ref v) => fmt_bytestring(v, fmt),
        Bencode::List(ref v) => {
            write!(fmt, "[")?;
            let mut first = true;
            for value in v.iter() {
                if first {
                    first = false;
                } else {
                    write!(fmt, ", ")?;
                }
                write!(fmt, "{}", *value)?;
            }
            write!(fmt, "]")
        }
        Bencode::Dict(ref v) => {
            write!(fmt, "{{")?;
            let mut first = true;
            for (key, value) in v.iter() {
                if first {
                    first = false;
                } else {
                    write!(fmt, ", ")?;
                }
                write!(fmt, "{}: {}", *key, *value)?;
            }
            write!(fmt, "}}")
        }
    }
}

pub type ListVec = Vec<Bencode>;
pub type DictMap = BTreeMap<util::ByteString, Bencode>;

impl Bencode {
    pub fn to_writer<W: io::Write>(&self, writer: W) ->
        Result<<&'_ mut Encoder<W> as serde::Serializer>::Ok,
               <&'_ mut Encoder<W> as serde::Serializer>::Error> {
        let mut encoder = Encoder::new(writer);
        self.serialize(&mut encoder)
    }

    pub fn to_bytes(&self) -> EncoderResult<Vec<u8>> {
        let mut writer = vec![];
        Ok(self.to_writer(&mut writer).and(Ok(writer))?)
    }
}

impl serde::Serialize for Bencode {
    fn serialize<S: serde::Serializer>(&self, e: S) -> Result<S::Ok, S::Error> {
        match self {
            // Not a backward compatible, as rust-serialize version
            // did nothing, returning Ok(()); but we cannot do nothing
            // with Serde as we have tor return S::Ok somehow.
            &Bencode::Empty => e.serialize_none(),
            &Bencode::Number(v) => e.serialize_i64(v),
            &Bencode::ByteString(ref v) => e.serialize_str(unsafe { str::from_utf8_unchecked(v) }),
            &Bencode::List(ref v) => v.serialize(e),
            &Bencode::Dict(ref v) => v.serialize(e)
        }
    }
}

pub trait ToBencode {
    fn to_bencode(&self) -> Bencode;
}

pub trait FromBencode where Self: Sized {
    type Err;

    fn from_bencode(&Bencode) -> Result<Self, Self::Err>;
}

impl ToBencode for () {
    fn to_bencode(&self) -> Bencode {
        Bencode::ByteString(Vec::new())
    }
}

impl FromBencode for () {
    type Err = ();

    fn from_bencode(bencode: &Bencode) -> Result<(), ()> {
        match bencode {
            &Bencode::ByteString(ref v) => {
                if v.len() == 0 {
                    Ok(())
                } else {
                    Err(())
                }
            }
            _ => Err(())
        }
    }
}

impl<T: ToBencode> ToBencode for Option<T> {
    fn to_bencode(&self) -> Bencode {
        match self {
            &Some(ref v) => v.to_bencode(),
            &None => Bencode::ByteString(b"nil".to_vec())
        }
    }
}

impl<T: FromBencode> FromBencode for Option<T> {
    type Err = T::Err;

    fn from_bencode(bencode: &Bencode) -> Result<Option<T>, T::Err> {
        match bencode {
            &Bencode::ByteString(ref v) => {
                if v == b"nil" {
                    return Ok(None)
                }
            }
            _ => ()
        }
        FromBencode::from_bencode(bencode).map(|v| Some(v))
    }
}
macro_rules! derive_num_to_bencode(($t:ty) => (
    impl ToBencode for $t {
        fn to_bencode(&self) -> Bencode { Bencode::Number(*self as i64) }
    }
));

#[derive(Debug)]
pub enum NumFromBencodeError {
    OutOfRange(i64),
    InvalidType,
}

macro_rules! derive_num_from_bencode(($t:ty) => (
    impl FromBencode for $t {
        type Err = NumFromBencodeError;

        fn from_bencode(bencode: &Bencode) -> Result<$t, NumFromBencodeError> {
            match bencode {
                &Bencode::Number(v) => match FromPrimitive::from_i64(v) {
                    Some(n) => Ok(n),
                    None    => Err(NumFromBencodeError::OutOfRange(v)),
                },
                _ => Err(NumFromBencodeError::InvalidType),
            }
        }
    }
));

derive_num_to_bencode!(isize);
derive_num_from_bencode!(isize);

derive_num_to_bencode!(i8);
derive_num_from_bencode!(i8);

derive_num_to_bencode!(i16);
derive_num_from_bencode!(i16);

derive_num_to_bencode!(i32);
derive_num_from_bencode!(i32);

derive_num_to_bencode!(i64);
derive_num_from_bencode!(i64);

derive_num_to_bencode!(usize);
derive_num_from_bencode!(usize);

derive_num_to_bencode!(u8);
derive_num_from_bencode!(u8);

derive_num_to_bencode!(u16);
derive_num_from_bencode!(u16);

derive_num_to_bencode!(u32);
derive_num_from_bencode!(u32);

derive_num_to_bencode!(u64);
derive_num_from_bencode!(u64);

impl ToBencode for f32 {
    fn to_bencode(&self) -> Bencode {
        let mut buf: Vec<u8> = Vec::with_capacity(size_of::<f32>());
        buf.write_f32::<BigEndian>(*self).unwrap();
        Bencode::ByteString(buf)
    }
}

#[derive(Debug)]
pub enum FloatFromBencodeError {
    InvalidLen(usize),
    InvalidType,
}

impl FromBencode for f32 {
    type Err = FloatFromBencodeError;

    fn from_bencode(bencode: &Bencode) -> Result<f32, FloatFromBencodeError> {
        use FloatFromBencodeError::*;
        match bencode {
            &Bencode::ByteString(ref v)  => {
                let len = v.len();
                match len == size_of::<f32>() {
                  true  => Ok(Cursor::new(&v[..]).read_f32::<BigEndian>().unwrap()),
                  false => Err(InvalidLen(len)),
                }
            }
            _ => Err(InvalidType),
        }
    }
}

impl ToBencode for f64 {
    fn to_bencode(&self) -> Bencode {
        let mut buf: Vec<u8> = Vec::with_capacity(size_of::<f64>());
        buf.write_f64::<BigEndian>(*self).unwrap();
        Bencode::ByteString(buf)
    }
}

impl FromBencode for f64 {
    type Err = FloatFromBencodeError;

    fn from_bencode(bencode: &Bencode) -> Result<f64, FloatFromBencodeError> {
        use FloatFromBencodeError::*;
        match bencode {
            &Bencode::ByteString(ref v)  => {
                let len = v.len();
                match len == size_of::<f64>() {
                  true  => Ok(Cursor::new(&v[..]).read_f64::<BigEndian>().unwrap()),
                  false => Err(InvalidLen(len)),
                }
            }
            _ => Err(InvalidType),
        }
    }
}

impl ToBencode for bool {
    fn to_bencode(&self) -> Bencode {
        if *self {
            Bencode::ByteString(b"true".to_vec())
        } else {
            Bencode::ByteString(b"false".to_vec())
        }
    }
}

#[derive(Debug)]
pub enum BoolFromBencodeError {
    NotAString,
    InvalidString(Vec<u8>),
}

impl FromBencode for bool {
    type Err = BoolFromBencodeError;

    fn from_bencode(bencode: &Bencode) -> Result<bool, BoolFromBencodeError> {
        match bencode {
            &Bencode::ByteString(ref v) => {
                if v == b"true" {
                    Ok(true)
                } else if v == b"false" {
                    Ok(false)
                } else {
                    Err(BoolFromBencodeError::InvalidString(v.clone()))
                }
            }
            _ => Err(BoolFromBencodeError::NotAString)
        }
    }
}

impl ToBencode for char {
    fn to_bencode(&self) -> Bencode {
        Bencode::ByteString(self.to_string().as_bytes().to_vec())
    }
}

#[derive(Debug)]
pub enum CharFromBencodeError {
    FromUtf8(Utf8Error),
    EmptyString,
    MultipleChars,
    InvalidType,
}

impl FromBencode for char {
    type Err = CharFromBencodeError;

    fn from_bencode(bencode: &Bencode) -> Result<char, CharFromBencodeError> {
        let s: Result<String, StringFromBencodeError> = FromBencode::from_bencode(bencode);
        match s {
            Ok(s) => {
                let mut it = s.chars();
                match it.next() {
                    None  => Err(CharFromBencodeError::EmptyString),
                    Some(c) => match it.next() {
                        None    => Ok(c),
                        Some(_) => Err(CharFromBencodeError::MultipleChars),
                    }
                }
            },
            Err(e)  => match e {
                StringFromBencodeError::FromUtf8(e) => Err(CharFromBencodeError::FromUtf8(e)),
                StringFromBencodeError::InvalidType => Err(CharFromBencodeError::InvalidType),
            }
        }
    }
}

impl ToBencode for String {
    fn to_bencode(&self) -> Bencode { Bencode::ByteString(self.as_bytes().to_vec()) }
}

#[derive(Debug)]
pub enum StringFromBencodeError {
    FromUtf8(Utf8Error),
    InvalidType,
}

impl FromBencode for String {
    type Err = StringFromBencodeError;

    fn from_bencode(bencode: &Bencode) -> Result<String, StringFromBencodeError> {
        use StringFromBencodeError::*;
        match bencode {
            &Bencode::ByteString(ref v) => std::str::from_utf8(v).map(|s| s.to_string()).map_err(FromUtf8),
            _ => Err(InvalidType),
        }
    }
}

impl<T: ToBencode> ToBencode for Vec<T> {
    fn to_bencode(&self) -> Bencode { Bencode::List(self.iter().map(|e| e.to_bencode()).collect()) }
}

#[derive(Debug)]
pub enum VecFromBencodeError<E> {
    Underlying(E),
    InvalidType,
}

impl<T: FromBencode> FromBencode for Vec<T> {
    type Err = VecFromBencodeError<T::Err>;

    fn from_bencode(bencode: &Bencode) -> Result<Vec<T>, VecFromBencodeError<T::Err>> {
        match bencode {
            &Bencode::List(ref es) => {
                let mut list = Vec::new();
                for e in es.iter() {
                    match FromBencode::from_bencode(e) {
                        Ok(v) => list.push(v),
                        Err(e) => return Err(VecFromBencodeError::Underlying(e)),
                    }
                }
                Ok(list)
            }
            _ => Err(VecFromBencodeError::InvalidType),
        }
    }
}

macro_rules! map_to_bencode {
    ($m:expr) => {{
        let mut m = BTreeMap::new();
        for (key, value) in $m.iter() {
            m.insert(util::ByteString::from_vec(key.as_bytes().to_vec()), value.to_bencode());
        }
        Bencode::Dict(m)
    }}
}

#[derive(Debug)]
pub enum MapFromBencodeError<E> {
    Underlying(E),
    KeyInvalidUtf8(Utf8Error),
    InvalidType,
}

macro_rules! map_from_bencode {
    ($mty:ident, $bencode:expr) => {{
        let res = match $bencode {
            &Bencode::Dict(ref map) => {
                let mut m = $mty::new();
                for (key, value) in map.iter() {
                    match str::from_utf8(key.as_slice()) {
                        Ok(k) => {
                            let val: Result<T, T::Err> = FromBencode::from_bencode(value);
                            match val {
                                Ok(v) => m.insert(k.to_string(), v),
                                Err(e) => return Err(MapFromBencodeError::Underlying(e)),
                            }
                        }
                        Err(e) => return Err(MapFromBencodeError::KeyInvalidUtf8(e)),
                    };
                }
                Ok(m)
            }
            _ => Err(MapFromBencodeError::InvalidType),
        };
        res
    }}
}

impl<T: ToBencode> ToBencode for BTreeMap<String, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for BTreeMap<String, T> {
    type Err = MapFromBencodeError<T::Err>;
    fn from_bencode(bencode: &Bencode) -> Result<BTreeMap<String, T>, MapFromBencodeError<T::Err>> {
        map_from_bencode!(BTreeMap, bencode)
    }
}

impl<T: ToBencode> ToBencode for HashMap<String, T> {
    fn to_bencode(&self) -> Bencode {
        map_to_bencode!(self)
    }
}

impl<T: FromBencode> FromBencode for HashMap<String, T> {
    type Err = MapFromBencodeError<T::Err>;
    fn from_bencode(bencode: &Bencode) -> Result<HashMap<String, T>, MapFromBencodeError<T::Err>> {
        map_from_bencode!(HashMap, bencode)
    }
}

pub fn from_buffer(buf: &[u8]) -> Result<Bencode, Error> {
    from_iter(buf.iter().map(|b| *b))
}

pub fn from_vec(buf: Vec<u8>) -> Result<Bencode, Error> {
    from_buffer(&buf[..])
}

pub fn from_iter<T: Iterator<Item=u8>>(iter: T) -> Result<Bencode, Error> {
    let streaming_parser = StreamingParser::new(iter);
    let mut parser = Parser::new(streaming_parser);
    parser.parse()
}

pub fn encode<T: serde::Serialize>(t: T) -> EncoderResult<Vec<u8>> {
    let mut w = vec![];
    let mut encoder = Encoder::new(&mut w);
    t.serialize(&mut encoder).and(Ok(w))
}

pub struct Encoder<W: io::Write> {
    writer: W,
    writers: Vec<Vec<u8>>,
    is_none: bool,
    stack: Vec<BTreeMap<util::ByteString, Vec<u8>>>,
}

impl<W: io::Write> Encoder<W> {
    pub fn new(writer: W) -> Encoder<W> {
        Encoder {
            writer: writer,
            writers: Vec::new(),
            is_none: false,
            stack: Vec::new()
        }
    }

    fn get_writer(&mut self) -> &mut dyn io::Write {
        if self.writers.len() == 0 {
            &mut self.writer as &mut dyn io::Write
        } else {
            self.writers.last_mut().unwrap() as &mut dyn io::Write
        }
    }

    pub fn into_inner(self) -> W {
        if self.writers.len() == 0 {
            self.writer
        } else {
            // Invariant is: into_inner is called only by user code when Encoder::encode call
            // is complete, and when it is complete, self.writers is empty.
            panic!(
                "Destroying unflushed bencode Encoder.  Shouldn't happen, some invariant is broken."
            );
        }
    }

    fn encode_dict<'b, 'c>(&'b mut self, dict: &'c BTreeMap<util::ByteString, Vec<u8>>) -> EncoderResult<()> {
        write!(self.get_writer(), "d")?;
        for (key, value) in dict.iter() {
            key.serialize(& mut *self)?;
            self.get_writer().write_all(value)?;
        }
        write!(self.get_writer(), "e")?;
        Ok(())
    }

    fn encode_bytestring(&mut self, v: &[u8]) -> EncoderResult<()> {
        write!(self.get_writer(), "{}:", v.len())?;
        Ok(self.get_writer().write_all(v)?)
    }

    fn error<T>(&mut self, msg: &'static str) -> EncoderResult<T> {
        Err(io::Error::new(io::ErrorKind::InvalidInput, msg).into())
    }
}


#[derive(Debug)]
pub enum SerializeErr {
    Io(io::Error),
    Msg(String),
}

impl fmt::Display for SerializeErr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{:?}", self)
    }
}

impl std::error::Error for SerializeErr {
}

impl From<io::Error> for SerializeErr {
    fn from(e: io::Error) -> Self {
        SerializeErr::Io(e)
    }
}

impl ser::Error for SerializeErr {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        SerializeErr::Msg(msg.to_string())
    }
}

pub type EncoderResult<T> = Result<T, SerializeErr>;

impl<'a, W: io::Write> serde::Serializer for &'a mut Encoder<W> {
    type Ok = ();
    type Error = SerializeErr;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeMap = SerializeMap<'a, W>;
    type SerializeStruct = Self;
    type SerializeTupleStruct = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeTupleVariant = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeStructVariant = ser::Impossible<Self::Ok, Self::Error>;

    fn serialize_unit(self) -> EncoderResult<Self::Ok> {
        Ok(write!(self.get_writer(), "0:")?)
    }

    fn serialize_none(self) -> EncoderResult<()> {
        self.is_none = true;
        Ok(write!(self.get_writer(), "3:nil")?)
    }

    fn serialize_some<T: ?Sized + serde::Serialize>(self, value: &T) -> EncoderResult<Self::Ok> {
        value.serialize(self)
    }

    fn serialize_u8(self, v: u8) -> EncoderResult<Self::Ok> { self.serialize_i64(v as i64) }

    fn serialize_u16(self, v: u16) -> EncoderResult<Self::Ok> { self.serialize_i64(v as i64) }

    fn serialize_u32(self, v: u32) -> EncoderResult<Self::Ok> { self.serialize_i64(v as i64) }

    fn serialize_u64(self, v: u64) -> EncoderResult<Self::Ok> {
        Ok(write!(self.get_writer(), "i{}e", v)?)
    }

    fn serialize_i8(self, v: i8) -> EncoderResult<Self::Ok> { self.serialize_i64(v as i64) }

    fn serialize_i16(self, v: i16) -> EncoderResult<Self::Ok> { self.serialize_i64(v as i64) }

    fn serialize_i32(self, v: i32) -> EncoderResult<Self::Ok> { self.serialize_i64(v as i64) }

    fn serialize_i64(self, v: i64) -> EncoderResult<Self::Ok> {
        Ok(write!(self.get_writer(), "i{}e", v)?)
    }

    fn serialize_bool(self, v: bool) -> EncoderResult<Self::Ok> {
        if v {
            self.serialize_str("true")
        } else {
            self.serialize_str("false")
        }
    }

    fn serialize_f32(self, v: f32) -> EncoderResult<Self::Ok> {
        let mut buf = Vec::with_capacity(4);
        buf.write_f32::<BigEndian>(v).unwrap();
        self.encode_bytestring(&buf[..])
    }

    fn serialize_f64(self, v: f64) -> EncoderResult<Self::Ok> {
        let mut buf = Vec::with_capacity(8);
        buf.write_f64::<BigEndian>(v).unwrap();
        self.encode_bytestring(&buf[..])
    }

    fn serialize_char(self, v: char) -> EncoderResult<Self::Ok> {
        self.encode_bytestring(&v.to_string().as_bytes())
    }

    fn serialize_str(self, v: &str) -> EncoderResult<Self::Ok> {
        self.encode_bytestring(v.as_bytes())
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T) -> EncoderResult<Self::Ok> {
        self.error("serialize_newtype_variant not implemented")
    }

    fn serialize_bytes(self, v: &[u8]) -> EncoderResult<Self::Ok> {
        self.encode_bytestring(v)
    }

    fn serialize_seq(self, _len: Option<usize>) -> EncoderResult<Self::SerializeSeq> {
        write!(self.get_writer(), "l")?;  // SerializeSeq::end will write "e"
        Ok(self)
    }

    fn serialize_tuple(self, _len: usize) -> EncoderResult<Self::SerializeTuple> {
        self.error("serialize_tuple not implemented")
    }

    fn serialize_map(self, _size: Option<usize>) -> EncoderResult<Self::SerializeMap> {
        // bencode requires keys to be sorted, thus we store key-values into btree.
        // TODO: hash+sort can be more efficient than btree.
        self.stack.push(BTreeMap::new());
        Ok(SerializeMap::new(self))
    }

    fn serialize_tuple_struct(self, _name: &'static str, _len: usize)
                              -> EncoderResult<Self::SerializeTupleStruct> {
        self.error("serialize_tuple_struct not implemented")
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.error("serialize_tuple_variant not implemented")
    }

    fn serialize_unit_struct(self, _name: &'static str) -> EncoderResult<Self::Ok> {
        self.serialize_unit()
    }

    fn serialize_unit_variant(self,
                              _name: &'static str,
                              _variant_index: u32,
                              _variant: &'static str) -> EncoderResult<Self::Ok> {
        self.error("serialize_unit_variant not implemented")
    }

    fn serialize_struct_variant(self,
                                _name: &'static str,
                                _variant_index: u32,
                                _variant: &'static str,
                                _len: usize,
    ) -> EncoderResult<Self::SerializeStructVariant> {
        self.error("serialize_struct_variant not implemented")
    }

    fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(
        self,
        _name: &'static str,
        value: &T
    ) -> EncoderResult<Self::Ok> {
        value.serialize(self)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize
    ) -> EncoderResult<Self::SerializeStruct> {
        // We serialize struct as a map {field_name: value}.
        self.stack.push(BTreeMap::new());
        Ok(self)
    }
}

impl<W: io::Write> ser::SerializeSeq for &mut Encoder<W> {
    type Ok = ();
    type Error = SerializeErr;

    fn serialize_element<'c, 'd, T: serde::Serialize + ?Sized>(&'c mut self, elt: &'d T) -> EncoderResult<()> {
        let selff = &mut **self;
        elt.serialize(selff)
    }

    fn end(self) -> EncoderResult<()> {
        self.is_none = false;
        Ok(write!(self.get_writer(), "e")?)
    }
}

impl<W: io::Write> ser::SerializeTuple for &mut Encoder<W> {
    type Ok = ();
    type Error = SerializeErr;

    fn serialize_element<T: serde::Serialize + ?Sized>(&mut self, _elt: &T) -> EncoderResult<()> {
        self.error("serialize_tuple not implemented")
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.error("serialize_tuple not implemented")
    }
}

struct SerializeKey ();

impl SerializeKey {
    fn expect_string<T>(self) -> EncoderResult<T> {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "Only 'string' map keys allowed").into())
    }
}

impl ser::Serializer for SerializeKey {
    type Ok = util::ByteString;
    type Error = SerializeErr;
    type SerializeSeq = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeTuple = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeMap = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeStruct = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeTupleStruct = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeTupleVariant = ser::Impossible<Self::Ok, Self::Error>;
    type SerializeStructVariant = ser::Impossible<Self::Ok, Self::Error>;

    fn serialize_str(self, v: &str) -> EncoderResult<Self::Ok> {
        self.serialize_bytes(v.as_bytes())
    }

    fn serialize_bytes(self, v: &[u8]) -> EncoderResult<Self::Ok> {
        Ok(util::ByteString::from_slice(v))
    }

    fn serialize_unit(self) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_none(self) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_some<T: ?Sized + serde::Serialize>(self, _value: &T) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_u8(self, _v: u8) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_u16(self, _v: u16) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_u32(self, _v: u32) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_u64(self, _v: u64) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_i8(self, _v: i8) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_i16(self, _v: i16) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_i32(self, _v: i32) -> EncoderResult<Self::Ok> {
        self.expect_string()

    }

    fn serialize_i64(self, _v: i64) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_bool(self, _v: bool) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_f32(self, _v: f32) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_f64(self, _v: f64) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_char(self, _v: char) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_seq(self, _len: Option<usize>) -> EncoderResult<Self::SerializeSeq> {
        self.expect_string()
    }

    fn serialize_tuple(self, _len: usize) -> EncoderResult<Self::SerializeTuple> {
        self.expect_string()
    }

    fn serialize_map(self, _size: Option<usize>) -> EncoderResult<Self::SerializeMap> {
        self.expect_string()
    }

    fn serialize_tuple_struct(self, _name: &'static str, _len: usize)
                              -> EncoderResult<Self::SerializeTupleStruct> {
        self.expect_string()
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize) -> EncoderResult<Self::SerializeTupleVariant> {
        self.expect_string()
    }

    fn serialize_unit_struct(self, _name: &'static str) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_unit_variant(self,
                              _name: &'static str,
                              _variant_index: u32,
                              _variant: &'static str) -> EncoderResult<Self::Ok> {
        self.expect_string()
    }

    fn serialize_struct_variant(self,
                                _name: &'static str,
                                _variant_index: u32,
                                _variant: &'static str,
                                _len: usize,
    ) -> EncoderResult<Self::SerializeStructVariant> {
        self.expect_string()
    }

    fn serialize_newtype_struct<T: ?Sized + serde::Serialize>(
        self,
        _name: &'static str,
        value: &T
    ) -> EncoderResult<Self::Ok> {
        value.serialize(self)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize
    ) -> EncoderResult<Self::SerializeStruct> {
        self.expect_string()
    }
}


pub struct SerializeMap<'a, W: io::Write> {
    parent: &'a mut Encoder<W>,
    key: Option<util::ByteString>,
    dict: BTreeMap<util::ByteString, Vec<u8>>
}

impl<'a, W: io::Write> SerializeMap<'a, W> {
    fn new(parent: &'a mut Encoder<W>) -> Self {
        Self {
            parent,
            key: None,
            dict: Default::default()
        }
    }
}

impl<'a, W: io::Write> ser::SerializeMap for SerializeMap<'a, W> {
    type Ok = ();
    type Error = SerializeErr;

    fn serialize_key<T: ?Sized>(&mut self, key: &T) -> EncoderResult<()>
    where T: serde::Serialize {
        self.key = Some(key.serialize(SerializeKey())?);
        Ok(())
    }

    fn serialize_value<T: ?Sized>(&mut self, val: &T) -> EncoderResult<()>
    where T: serde::Serialize {
        let mut data = vec![];
        let mut enc = Encoder::new(&mut data);
        val.serialize(&mut enc)?;
        let key = self.key.take().unwrap();
        self.dict.insert(key, data);
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.parent.encode_dict(&self.dict)
    }
}

// Struct is serialized as a map {field_name: value}.
impl<W: io::Write> ser::SerializeStruct for &mut Encoder<W> {
    type Ok = ();
    type Error = SerializeErr;

    fn serialize_field<T: ?Sized>(&mut self, key: &'static str, value: &T) -> EncoderResult<()>
    where T: serde::Serialize {
        self.writers.push(vec![]);
        value.serialize(&mut **self)?;

        let data = self.writers.pop().unwrap();
        let dict = self.stack.last_mut().unwrap();
        if !self.is_none {  // It seems fields with None values are just skipped, that makes sens.
            dict.insert(util::ByteString::from_slice(key.as_bytes()), data);
        }
        self.is_none = false;
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        let dict = self.stack.pop().unwrap();
        self.encode_dict(&dict)?;
        self.is_none = false;
        Ok(())
    }
}

pub struct Parser<T> {
    reader: T,
    depth: u32,
}

impl<T: Iterator<Item=BencodeEvent>> Parser<T> {
    pub fn new(reader: T) -> Parser<T> {
        Parser {
            reader: reader,
            depth: 0
        }
    }

    pub fn parse(&mut self) -> Result<Bencode, Error> {
        let next = self.reader.next();
        self.parse_elem(next)
    }

    fn parse_elem(&mut self, current: Option<BencodeEvent>) -> Result<Bencode, Error> {
        let res = match current {
            Some(NumberValue(v)) => Ok(Bencode::Number(v)),
            Some(ByteStringValue(v)) => Ok(Bencode::ByteString(v)),
            Some(ListStart) => self.parse_list(),
            Some(DictStart) => self.parse_dict(),
            Some(ParseError(err)) => Err(err),
            None => Ok(Empty),
            x => panic!("[root] Unreachable but got {:?}", x)
        };
        if self.depth == 0 {
            let next = self.reader.next();
            match res {
                Err(_) => res,
                _ => {
                    match next {
                        Some(ParseError(err)) => Err(err),
                        None => res,
                        x => panic!("Unreachable but got {:?}", x)
                    }
                }
            }
        } else {
            res
        }
    }

    fn parse_list(&mut self) -> Result<Bencode, Error> {
        self.depth += 1;
        let mut list = Vec::new();
        loop {
            let current = self.reader.next();
            match current {
                Some(ListEnd) => break,
                Some(ParseError(err)) => return Err(err),
                Some(_) => {
                    match self.parse_elem(current) {
                        Ok(v) => list.push(v),
                        err@Err(_) => return err
                    }
                }
                x => panic!("[list] Unreachable but got {:?}", x)
            }
        }
        self.depth -= 1;
        Ok(Bencode::List(list))
    }

    fn parse_dict(&mut self) -> Result<Bencode, Error> {
        self.depth += 1;
        let mut map = BTreeMap::new();
        loop {
            let mut current = self.reader.next();
            let key = match current {
                Some(DictEnd) => break,
                Some(DictKey(v)) => util::ByteString::from_vec(v),
                Some(ParseError(err)) => return Err(err),
                x => panic!("[dict] Unreachable but got {:?}", x)
            };
            current = self.reader.next();
            let value = self.parse_elem(current)?;
            map.insert(key, value);
        }
        self.depth -= 1;
        Ok(Bencode::Dict(map))
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum DecoderError {
    Message(String),
    StringEncoding(Vec<u8>),
    Expecting(&'static str, String),
    Unimplemented(&'static str),
}

impl fmt::Display for DecoderError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{:?}", self)
    }
}

impl de::StdError for DecoderError {
}

impl de::Error for DecoderError {
    fn custom<T>(msg: T) -> Self
    where T: fmt::Display {
        DecoderError::Message(msg.to_string())
    }
}

pub type DecoderResult<T> = Result<T, DecoderError>;

pub struct Decoder<'de> {
    value: &'de Bencode,
}

impl<'de> Decoder<'de> {
    pub fn new(bencode: &'de Bencode) -> Decoder<'de> {
        Decoder {
            value: bencode,
        }
    }

    fn try_read<T: FromBencode>(&mut self, ty: &'static str) -> DecoderResult<T> {
        match FromBencode::from_bencode(self.value).ok() {
            Some(v) => Ok(v),
            None => Err(Message(format!("Error decoding value as '{}': {:?}", ty, self.value)))
        }
    }

    fn unimplemented<T>(&self, m: &'static str) -> DecoderResult<T> {
        Err(Unimplemented(m))
    }

    fn error<T, M: fmt::Display>(&self, m: M) -> DecoderResult<T> {
        Err(<DecoderError as de::Error>::custom(m))
    }
}

impl<'de, 'a> serde::Deserializer<'de> for &'a mut Decoder<'de> {
    type Error = DecoderError;

    fn deserialize_any<V: de::Visitor<'de>>(self, _visitor: V) -> DecoderResult<V::Value> {
        self.unimplemented("deserialize_any not implemented")
    }

    fn deserialize_bool<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_bool(self.try_read("bool")?)
    }

    fn deserialize_u8<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_u8(self.try_read("u8")?)
    }

    fn deserialize_u16<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_u16(self.try_read("u16")?)
    }

    fn deserialize_u32<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_u32(self.try_read("u32")?)
    }

    fn deserialize_u64<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_u64(self.try_read("u64")?)
    }

    fn deserialize_i8<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_i8(self.try_read("i8")?)
    }

    fn deserialize_i16<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_i16(self.try_read("i16")?)
    }

    fn deserialize_i32<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_i32(self.try_read("i32")?)
    }

    fn deserialize_i64<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_i64(self.try_read("i64")?)
    }

    fn deserialize_f32<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_f32(self.try_read("f32")?)
    }

    fn deserialize_f64<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_f64(self.try_read("f64")?)
    }

    fn deserialize_char<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        visitor.visit_char(self.try_read("char")?)
    }

    fn deserialize_str<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        self.deserialize_string(visitor)
    }

    fn deserialize_string<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        match self.value {
            &Bencode::ByteString(ref v) => {
                String::from_utf8(v.clone()
                ).map_err(|err| StringEncoding(err.into_bytes())
                ).and_then(|v| visitor.visit_string(v)
                )
            }
            _ => self.error(&format!("Error decoding value as str: {:?}", self.value))
        }
    }

    fn deserialize_bytes<V: de::Visitor<'de>>(self, _visitor: V) -> DecoderResult<V::Value> {
        Err(DecoderError::Unimplemented("deserialize_bytes not implemented"))
    }

    fn deserialize_byte_buf<V: de::Visitor<'de>>(self, _visitor: V) -> DecoderResult<V::Value> {
        Err(DecoderError::Unimplemented("deserialize_byte_buf not implemented"))
    }

    fn deserialize_option<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        match self.value {
            &Bencode::Empty => visitor.visit_none(),
            &Bencode::ByteString(ref v) => {
                if v == b"nil" {
                    visitor.visit_none()
                } else {
                    visitor.visit_some(self)
                }
            },
            _ => {
                visitor.visit_some(self)
            }
        }
    }

    fn deserialize_unit<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        self.try_read("unit")?;
        visitor.visit_unit()
    }

    fn deserialize_unit_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V
    ) -> DecoderResult<V::Value> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_enum<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        _visitor: V
    ) -> DecoderResult<V::Value> {
        self.unimplemented("deserialize_enum")
    }
    
    fn deserialize_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V
    ) -> DecoderResult<V::Value> {
        self.deserialize_map(visitor)
    }

    fn deserialize_newtype_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        _visitor: V
    ) -> DecoderResult<V::Value> {
        self.unimplemented("deserialize_newtype_struct")
    }

    fn deserialize_seq<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        match self.value {
            &Bencode::List(ref list) => {
                visitor.visit_seq(SeqDeserializer {
                    vals: list.iter(),
                })
            }
            val => Err(Expecting("List", val.to_string()))
        }
    }

    fn deserialize_tuple<V: de::Visitor<'de>>(
        self,
        _len: usize,
        _visitor: V
    ) -> DecoderResult<V::Value> {
        self.unimplemented("deserialize_tuple")
    }

    fn deserialize_tuple_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        _visitor: V
    ) -> DecoderResult<V::Value> {
        self.unimplemented("deserialize_tuple_struct")
    }

    fn deserialize_map<V: de::Visitor<'de>>(
        self,
        visitor: V
    ) -> DecoderResult<V::Value> {
        match self.value {
            &Bencode::Dict(ref m) => {
                visitor.visit_map(MapDeserializer {
                    keys: m.keys(),
                    vals: m.values()
                })
            }
            val => Err(Expecting("Dict", val.to_string()))
        }
    }
    
    fn deserialize_identifier<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        self.deserialize_string(visitor)
    }

    fn deserialize_ignored_any<V: de::Visitor<'de>>(self, _visitor: V) -> DecoderResult<V::Value> {
        self.unimplemented("deserialize_ignored_any")
    }
}

struct MapDeserializer<'de> {
    keys: std::collections::btree_map::Keys<'de, util::ByteString, Bencode>,
    vals: std::collections::btree_map::Values<'de, util::ByteString, Bencode>,
}

impl<'de> de::MapAccess<'de> for MapDeserializer<'de> {
    type Error = DecoderError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error> where
        K: de::DeserializeSeed<'de> {
        if let Some(key) = self.keys.next() {
            seed.deserialize(KeyDeserializer {
                key,
            }).map(Some)
        } else {
            Ok(None)
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error> where
        V: de::DeserializeSeed<'de> {
        seed.deserialize(&mut Decoder {
            // We use there unwrap because length of vals is equal to length of keys,
            // and sane client code calls next_value_seed only if next_key_seed
            // returns Some.  It is definitely true for autogenerated code, but not
            // for manual code.
            value: self.vals.next().unwrap()
        })
    }
}

struct SeqDeserializer<'de> {
    vals: std::slice::Iter<'de, Bencode>,
}

impl<'de> de::SeqAccess<'de> for SeqDeserializer<'de> {
    type Error = DecoderError;

    fn next_element_seed<V>(&mut self, seed: V) -> Result<Option<V::Value>, Self::Error> where
        V: de::DeserializeSeed<'de> {
        match self.vals.next() {
            Some(value) =>
                seed.deserialize(&mut Decoder {
                    value
                }).map(Some),
            None => Ok(None)
        }
    }
}

struct KeyDeserializer<'de> {
    key: &'de util::ByteString,
}

impl<'de> serde::Deserializer<'de> for KeyDeserializer<'de> {
    type Error = DecoderError;

    fn deserialize_any<V: de::Visitor<'de>>(self, visitor: V) -> DecoderResult<V::Value> {
        match String::from_utf8(self.key.as_slice().to_vec()) {
            Ok(s) => visitor.visit_string(s),
            Err(err) => Err(StringEncoding(err.into_bytes()))
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> DecoderResult<V::Value>
    where
        V: de::Visitor<'de>,
    {
        // Map keys cannot be null.
        visitor.visit_some(self)
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 bytes byte_buf newtype_struct enum
        f32 f64 char str string unit unit_struct seq tuple tuple_struct map
        struct identifier ignored_any
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashMap;

    use serde::{Deserialize, Serialize};

    use streaming::Error;
    use streaming::BencodeEvent;
    use streaming::BencodeEvent::{NumberValue, ByteStringValue, ListStart,
                                  ListEnd, DictStart, DictKey, DictEnd, ParseError};

    use super::{Bencode, ToBencode};
    use super::{Parser, Decoder, DecoderResult, encode};

    use super::util;

    macro_rules! assert_encoding(($value:expr, $expected:expr) => ({
        let value = $value;
        let encoded = match encode(&value) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        assert_eq!($expected, encoded);
    }));

    macro_rules! assert_decoding(($enc:expr, $value:expr) => ({
        let bencode = super::from_vec($enc).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Deserialize::deserialize(&mut decoder);
        assert_eq!(Ok($value), result);
    }));

    macro_rules! gen_encode_test(($name:ident, $(($val:expr) -> $enc:expr),+) => {
        #[test]
        fn $name() {
            $(assert_encoding!($val, $enc);)+
        }
    });

    macro_rules! gen_tobencode_test(($name:ident, $(($val:expr) -> $enc:expr),+) => {
        #[test]
        fn $name() {
            $({
                let value = $val.to_bencode();
                assert_encoding!(value, $enc)
            };)+
        }
    });

    macro_rules! assert_identity(($value:expr) => ({
        let value = $value;
        let encoded = match encode(&value) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        let bencode = super::from_vec(encoded).unwrap();
        let mut decoder = Decoder::new(&bencode);
        let result = Deserialize::deserialize(&mut decoder);
        assert_eq!(Ok(value), result);
    }));

    macro_rules! gen_identity_test(($name:ident, $($val:expr),+) => {
        #[test]
        fn $name() {
            $(assert_identity!($val);)+
        }
    });

    macro_rules! gen_encode_identity_test(($name_enc:ident, $name_ident:ident, $(($val:expr) -> $enc:expr),+) => {
        gen_encode_test!($name_enc, $(($val) -> $enc),+);
        gen_identity_test!($name_ident, $($val),+);
    });

    macro_rules! gen_complete_test(($name_enc:ident, $name_benc:ident, $name_ident:ident, $(($val:expr) -> $enc:expr),+) => {
        gen_encode_test!($name_enc, $(($val) -> $enc),+);
        gen_tobencode_test!($name_benc, $(($val) -> $enc),+);
        gen_identity_test!($name_ident, $($val),+);
    });

    fn bytes(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    gen_complete_test!(encodes_unit,
                       tobencode_unit,
                       identity_unit,
                       (()) -> bytes("0:"));

    gen_complete_test!(encodes_option_none,
                       tobencode_option_none,
                       identity_option_none,
                       ({
                           let none: Option<isize> = None;
                           none
                       }) -> bytes("3:nil"));

    gen_complete_test!(encodes_option_some,
                       tobencode_option_some,
                       identity_option_some,
                       (Some(1isize)) -> bytes("i1e"),
                       (Some("rust".to_string())) -> bytes("4:rust"),
                       (Some(vec![(), ()])) -> bytes("l0:0:e"));

    gen_complete_test!(encodes_nested_option,
                       tobencode_nested_option,
                       identity_nested_option,
                       (Some(Some(1isize))) -> bytes("i1e"),
                       (Some(Some("rust".to_string()))) -> bytes("4:rust"));

    #[test]
    fn option_is_none_if_any_nested_option_is_none() {
        let value: Option<Option<isize>> = Some(None);
        let encoded = match encode(&value) {
            Ok(e) => e,
            Err(err) => panic!("Unexpected failure: {}", err)
        };
        let none: Option<Option<isize>> = None;
        assert_decoding!(encoded, none);
    }

    gen_complete_test!(encodes_zero_isize,
                       tobencode_zero_isize,
                       identity_zero_isize,
                       (0isize) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_isize,
                       tobencode_positive_isize,
                       identity_positive_isize,
                       (5isize) -> bytes("i5e"),
                       (99isize) -> bytes("i99e"),
                       (::std::isize::MAX) -> bytes(&format!("i{}e", ::std::isize::MAX)[..]));

    gen_complete_test!(encodes_negative_isize,
                       tobencode_negative_isize,
                       identity_negative_isize,
                       (-5isize) -> bytes("i-5e"),
                       (-99isize) -> bytes("i-99e"),
                       (::std::isize::MIN) -> bytes(&format!("i{}e", ::std::isize::MIN)[..]));

    gen_complete_test!(encodes_zero_i8,
                       tobencode_zero_i8,
                       identity_zero_i8,
                       (0i8) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i8,
                       tobencode_positive_i8,
                       identity_positive_i8,
                       (5i8) -> bytes("i5e"),
                       (99i8) -> bytes("i99e"),
                       (::std::i8::MAX) -> bytes(&format!("i{}e", ::std::i8::MAX)[..]));

    gen_complete_test!(encodes_negative_i8,
                       tobencode_negative_i8,
                       identity_negative_i8,
                       (-5i8) -> bytes("i-5e"),
                       (-99i8) -> bytes("i-99e"),
                       (::std::i8::MIN) -> bytes(&format!("i{}e", ::std::i8::MIN)[..]));

    gen_complete_test!(encodes_zero_i16,
                       tobencode_zero_i16,
                       identity_zero_i16,
                       (0i16) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i16,
                       tobencode_positive_i16,
                       identity_positive_i16,
                       (5i16) -> bytes("i5e"),
                       (99i16) -> bytes("i99e"),
                       (::std::i16::MAX) -> bytes(&format!("i{}e", ::std::i16::MAX)[..]));

    gen_complete_test!(encodes_negative_i16,
                       tobencode_negative_i16,
                       identity_negative_i16,
                       (-5i16) -> bytes("i-5e"),
                       (-99i16) -> bytes("i-99e"),
                       (::std::i16::MIN) -> bytes(&format!("i{}e", ::std::i16::MIN)[..]));

    gen_complete_test!(encodes_zero_i32,
                       tobencode_zero_i32,
                       identity_zero_i32,
                       (0i32) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i32,
                       tobencode_positive_i32,
                       identity_positive_i32,
                       (5i32) -> bytes("i5e"),
                       (99i32) -> bytes("i99e"),
                       (::std::i32::MAX) -> bytes(&format!("i{}e", ::std::i32::MAX)[..]));

    gen_complete_test!(encodes_negative_i32,
                       tobencode_negative_i32,
                       identity_negative_i32,
                       (-5i32) -> bytes("i-5e"),
                       (-99i32) -> bytes("i-99e"),
                       (::std::i32::MIN) -> bytes(&format!("i{}e", ::std::i32::MIN)[..]));

    gen_complete_test!(encodes_zero_i64,
                       tobencode_zero_i64,
                       identity_zero_i64,
                       (0i64) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_i64,
                       tobencode_positive_i64,
                       identity_positive_i64,
                       (5i64) -> bytes("i5e"),
                       (99i64) -> bytes("i99e"),
                       (::std::i64::MAX) -> bytes(&format!("i{}e", ::std::i64::MAX)[..]));

    gen_complete_test!(encodes_negative_i64,
                       tobencode_negative_i64,
                       identity_negative_i64,
                       (-5i64) -> bytes("i-5e"),
                       (-99i64) -> bytes("i-99e"),
                       (::std::i64::MIN) -> bytes(&format!("i{}e", ::std::i64::MIN)[..]));

    gen_complete_test!(encodes_zero_usize,
                       tobencode_zero_usize,
                       identity_zero_usize,
                       (0usize) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_usize,
                       tobencode_positive_usize,
                       identity_positive_usize,
                       (5usize) -> bytes("i5e"),
                       (99usize) -> bytes("i99e"),
                       (::std::usize::MAX / 2) -> bytes(&format!("i{}e", ::std::usize::MAX / 2)[..]));

    gen_complete_test!(encodes_zero_u8,
                       tobencode_zero_u8,
                       identity_zero_u8,
                       (0u8) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u8,
                       tobencode_positive_u8,
                       identity_positive_u8,
                       (5u8) -> bytes("i5e"),
                       (99u8) -> bytes("i99e"),
                       (::std::u8::MAX) -> bytes(&format!("i{}e", ::std::u8::MAX)[..]));

    gen_complete_test!(encodes_zero_u16,
                       tobencode_zero_u16,
                       identity_zero_u16,
                       (0u16) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u16,
                       tobencode_positive_u16,
                       identity_positive_u16,
                       (5u16) -> bytes("i5e"),
                       (99u16) -> bytes("i99e"),
                       (::std::u16::MAX) -> bytes(&format!("i{}e", ::std::u16::MAX)[..]));

    gen_complete_test!(encodes_zero_u32,
                       tobencode_zero_u32,
                       identity_zero_u32,
                       (0u32) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u32,
                       tobencode_positive_u32,
                       identity_positive_u32,
                       (5u32) -> bytes("i5e"),
                       (99u32) -> bytes("i99e"),
                       (::std::u32::MAX) -> bytes(&format!("i{}e", ::std::u32::MAX)[..]));

    gen_complete_test!(encodes_zero_u64,
                       tobencode_zero_u64,
                       identity_zero_u64,
                       (0u64) -> bytes("i0e"));

    gen_complete_test!(encodes_positive_u64,
                       tobencode_positive_u64,
                       identity_positive_u64,
                       (5u64) -> bytes("i5e"),
                       (99u64) -> bytes("i99e"),
                       (::std::u64::MAX / 2) -> bytes(&format!("i{}e", ::std::u64::MAX / 2)[..]));

    gen_complete_test!(encodes_bool,
                       tobencode_bool,
                       identity_bool,
                       (true) -> bytes("4:true"),
                       (false) -> bytes("5:false"));

    gen_complete_test!(encodes_zero_f32,
                       tobencode_zero_f32,
                       identity_zero_f32,
                       (0.0f32) -> vec![b'4', b':', 0, 0, 0, 0]);

    gen_complete_test!(encodes_positive_f32,
                       tobencode_positive_f32,
                       identity_positive_f32,
                       (99.0f32) -> vec![b'4', b':', 0x42, 0xc6, 0x0, 0x0],
                       (101.12345f32) -> vec![b'4', b':', 0x42, 0xca, 0x3f, 0x35]);

    gen_complete_test!(encodes_negative_f32,
                       tobencode_negative_f32,
                       identity_negative_f32,
                       (-99.0f32) -> vec![b'4', b':', 0xc2, 0xc6, 0, 0],
                       (-101.12345f32) -> vec![b'4', b':', 0xc2, 0xca, 0x3f, 0x35]);

    gen_complete_test!(encodes_zero_f64,
                       tobencode_zero_f64,
                       identity_zero_f64,
                       (0.0f64) -> vec![b'8', b':', 0, 0, 0, 0, 0, 0, 0, 0]);

    gen_complete_test!(encodes_positive_f64,
                       tobencode_positive_f64,
                       identity_positive_f64,
                       (99.0f64) -> vec![b'8', b':', 0x40, 0x58, 0xc0, 0x0, 0x0, 0x0, 0x0, 0x0],
                       (101.12345f64) -> vec![b'8', b':', 0x40, 0x59, 0x47, 0xe6, 0x9a, 0xd4, 0x2c, 0x3d]);

    gen_complete_test!(encodes_negative_f64,
                       tobencode_negative_f64,
                       identity_negative_f64,
                       (-99.0f64) -> vec![b'8', b':', 0xc0, 0x58, 0xc0, 0, 0, 0, 0, 0],
                       (-101.12345f64) -> vec![b'8', b':', 0xc0, 0x59, 0x47, 0xe6, 0x9a, 0xd4, 0x2c, 0x3d]);

    gen_complete_test!(encodes_lower_letter_char,
                       tobencode_lower_letter_char,
                       identity_lower_letter_char,
                       ('a') -> bytes("1:a"),
                       ('c') -> bytes("1:c"),
                       ('z') -> bytes("1:z"));

    gen_complete_test!(encodes_upper_letter_char,
                       tobencode_upper_letter_char,
                       identity_upper_letter_char,
                       ('A') -> bytes("1:A"),
                       ('C') -> bytes("1:C"),
                       ('Z') -> bytes("1:Z"));

    gen_complete_test!(encodes_multibyte_char,
                       tobencode_multibyte_char,
                       identity_multibyte_char,
                       ('ệ') -> bytes("3:ệ"),
                       ('虎') -> bytes("3:虎"));

    gen_complete_test!(encodes_control_char,
                       tobencode_control_char,
                       identity_control_char,
                       ('\n') -> bytes("1:\n"),
                       ('\r') -> bytes("1:\r"),
                       ('\0') -> bytes("1:\0"));

    gen_complete_test!(encode_empty_str,
                      tobencode_empty_str,
                      identity_empty_str,
                      ("".to_string()) -> bytes("0:"));

    gen_complete_test!(encode_str,
                      tobencode_str,
                      identity_str,
                      ("a".to_string()) -> bytes("1:a"),
                      ("foo".to_string()) -> bytes("3:foo"),
                      ("This is nice!?#$%".to_string()) -> bytes("17:This is nice!?#$%"));

    gen_complete_test!(encode_str_with_multibyte_chars,
                      tobencode_str_with_multibyte_chars,
                      identity_str_with_multibyte_chars,
                      ("Löwe 老虎 Léopard".to_string()) -> bytes("21:Löwe 老虎 Léopard"),
                      ("いろはにほへとちりぬるを".to_string()) -> bytes("36:いろはにほへとちりぬるを"));

    gen_complete_test!(encodes_empty_vec,
                       tobencode_empty_vec,
                       identity_empty_vec,
                       ({
                           let empty: Vec<u8> = Vec::new();
                           empty
                       }) -> bytes("le"));

    gen_complete_test!(encodes_nonmpty_vec,
                       tobencode_nonmpty_vec,
                       identity_nonmpty_vec,
                       (vec![0isize, 1isize, 3isize, 4isize]) -> bytes("li0ei1ei3ei4ee"),
                       (vec!["foo".to_string(), "b".to_string()]) -> bytes("l3:foo1:be"));

    gen_complete_test!(encodes_nested_vec,
                       tobencode_nested_vec,
                       identity_nested_vec,
                       (vec![vec![1isize], vec![2isize, 3isize], vec![]]) -> bytes("lli1eeli2ei3eelee"));

    #[derive(Eq, PartialEq, Debug, Serialize, Deserialize)]
    struct SimpleStruct {
        a: usize,
        b: Vec<String>,
    }

    #[derive(Eq, PartialEq, Debug, Serialize, Deserialize)]
    struct InnerStruct {
        field_one: (),
        list: Vec<usize>,
        abc: String
    }

    #[derive(Eq, PartialEq, Debug, Serialize, Deserialize)]
    struct OuterStruct {
        inner: Vec<InnerStruct>,
        is_true: bool
    }

    gen_encode_identity_test!(encodes_struct,
                              identity_struct,
                              (SimpleStruct {
                                  b: vec!["foo".to_string(), "baar".to_string()],
                                  a: 123
                              }) -> bytes("d1:ai123e1:bl3:foo4:baaree"),
                              (SimpleStruct {
                                  a: 1234567890,
                                  b: vec![]
                              }) -> bytes("d1:ai1234567890e1:blee"));

    gen_encode_identity_test!(encodes_nested_struct,
                              identity_nested_struct,
                              (OuterStruct {
                                  is_true: true,
                                  inner: vec![InnerStruct {
                                      field_one: (),
                                      list: vec![99usize, 5usize],
                                      abc: "rust".to_string()
                                  }, InnerStruct {
                                      field_one: (),
                                      list: vec![],
                                      abc: "".to_string()
                                  }]
                              }) -> bytes("d\
                                           5:inner\
                                             l\
                                               d\
                                                 3:abc4:rust\
                                                 9:field_one0:\
                                                 4:list\
                                                   l\
                                                     i99e\
                                                     i5e\
                                                   e\
                                               e\
                                               d\
                                                 3:abc0:\
                                                 9:field_one0:\
                                                 4:listle\
                                               e\
                                             e\
                                           7:is_true4:true\
                                          e"));

    macro_rules! map(($m:ident, $(($key:expr, $val:expr)),*) => {{
        let mut _m = $m::new();
        $(_m.insert($key, $val);)*
        _m
    }});

    gen_complete_test!(encodes_hashmap,
                       bencode_hashmap,
                       identity_hashmap,
                       (map!(HashMap, ("a".to_string(), 1isize))) -> bytes("d1:ai1ee"),
                       (map!(HashMap, ("foo".to_string(), "a".to_string()), ("bar".to_string(), "bb".to_string()))) -> bytes("d3:bar2:bb3:foo1:ae"));

    gen_complete_test!(encodes_nested_hashmap,
                       bencode_nested_hashmap,
                       identity_nested_hashmap,
                       (map!(HashMap, ("a".to_string(), map!(HashMap, ("foo".to_string(), 101isize), ("bar".to_string(), 102isize))))) -> bytes("d1:ad3:bari102e3:fooi101eee"));
    #[test]
    fn decode_error_on_wrong_map_key_type() {
        let benc = Bencode::Dict(map!(BTreeMap, (util::ByteString::from_vec(bytes("foo")), Bencode::ByteString(bytes("bar")))));
        let mut decoder = Decoder::new(&benc);
        let res: DecoderResult<BTreeMap<isize, String>> = Deserialize::deserialize(&mut decoder);
        assert!(res.is_err());
    }

    #[test]
    fn encode_error_on_wrong_map_key_type() {
        let m = map!(HashMap, (1isize, "foo"));
        let encoded = encode(&m);
        assert!(encoded.is_err())
    }

    #[test]
    fn encodes_struct_fields_in_sorted_order() {
        #[derive(Serialize)]
        struct OrderedStruct {
            z: isize,
            a: isize,
            ab: isize,
            aa: isize,
        }
        let s = OrderedStruct {
            z: 4,
            a: 1,
            ab: 3,
            aa: 2
        };
        assert_eq!(encode(&s).unwrap(), bytes("d1:ai1e2:aai2e2:abi3e1:zi4ee"));
    }

    #[derive(Serialize, Deserialize, Eq, PartialEq, Debug, Clone)]
    struct OptionalStruct {
        a: Option<isize>,
        b: isize,
        c: Option<Vec<Option<bool>>>,
    }

    #[derive(Serialize, Deserialize, Eq, PartialEq, Debug)]
    struct OptionalStructOuter {
        a: Option<OptionalStruct>,
        b: Option<isize>,
    }

    static OPT_STRUCT: OptionalStruct = OptionalStruct {
        a: None,
        b: 10,
        c: None
    };

    #[test]
    fn struct_option_none_fields_are_not_encoded() {
        assert_encoding!(OPT_STRUCT.clone(), bytes("d1:bi10ee"));
    }


    #[test]
    fn struct_options_not_present_default_to_none() {
        assert_decoding!(bytes("d1:bi10ee"), OPT_STRUCT.clone());
    }

    gen_encode_identity_test!(encodes_nested_struct_fields,
                              identity_nested_struct_field,
                              ({
                                  OptionalStructOuter {
                                      a: Some(OPT_STRUCT.clone()),
                                      b: None
                                  }
                              }) -> bytes("d1:ad1:bi10eee"),
                              ({
                                  let a = OptionalStruct {
                                      a: None,
                                      b: 10,
                                      c: Some(vec![Some(true), None])
                                  };
                                  OptionalStructOuter {
                                      a: Some(a),
                                      b: Some(99)
                                  }
                              }) -> bytes("d1:ad1:bi10e1:cl4:true3:nilee1:bi99ee"));

    fn try_bencode(bencode: Bencode) -> Vec<u8> {
        match bencode.to_bytes() {
            Ok(v) => v,
            Err(err) => panic!("Unexpected error: {}", err)
        }
    }

    #[test]
    fn encodes_empty_bytestring() {
        assert_eq!(try_bencode(Bencode::ByteString(Vec::new())), bytes("0:"));
    }

    #[test]
    fn encodes_nonempty_bytestring() {
        assert_eq!(try_bencode(Bencode::ByteString(b"abc".to_vec())), bytes("3:abc"));
        assert_eq!(try_bencode(Bencode::ByteString(vec![0, 1, 2, 3])), bytes("4:\x00\x01\x02\x03"));
    }

    #[test]
    fn encodes_empty_list() {
        assert_eq!(try_bencode(Bencode::List(Vec::new())), bytes("le"));
    }

    #[test]
    fn encodes_nonempty_list() {
        assert_eq!(try_bencode(Bencode::List(vec![Bencode::Number(1)])), bytes("li1ee"));
        assert_eq!(try_bencode(Bencode::List(vec![Bencode::ByteString("foobar".as_bytes().to_vec()),
                          Bencode::Number(-1)])), bytes("l6:foobari-1ee"));
    }

    #[test]
    fn encodes_nested_list() {
        assert_eq!(try_bencode(Bencode::List(vec![Bencode::List(vec![])])), bytes("llee"));
        let list = Bencode::List(vec![Bencode::Number(1988), Bencode::List(vec![Bencode::Number(2014)])]);
        assert_eq!(try_bencode(list), bytes("li1988eli2014eee"));
    }

    #[test]
    fn encodes_empty_dict() {
        assert_eq!(try_bencode(Bencode::Dict(BTreeMap::new())), bytes("de"));
    }

    #[test]
    fn encodes_dict_with_items() {
        let mut m = BTreeMap::new();
        m.insert(util::ByteString::from_str("k1"), Bencode::Number(1));
        assert_eq!(try_bencode(Bencode::Dict(m.clone())), bytes("d2:k1i1ee"));
        m.insert(util::ByteString::from_str("k2"), Bencode::ByteString(vec![0, 0]));
        assert_eq!(try_bencode(Bencode::Dict(m.clone())), bytes("d2:k1i1e2:k22:\0\0e"));
    }

    #[test]
    fn encodes_nested_dict() {
        let mut outer = BTreeMap::new();
        let mut inner = BTreeMap::new();
        inner.insert(util::ByteString::from_str("val"), Bencode::ByteString(vec![68, 0, 90]));
        outer.insert(util::ByteString::from_str("inner"), Bencode::Dict(inner));
        assert_eq!(try_bencode(Bencode::Dict(outer)), bytes("d5:innerd3:val3:D\0Zee"));
    }

    #[test]
    fn encodes_dict_fields_in_sorted_order() {
        let mut m = BTreeMap::new();
        m.insert(util::ByteString::from_str("z"), Bencode::Number(1));
        m.insert(util::ByteString::from_str("abd"), Bencode::Number(3));
        m.insert(util::ByteString::from_str("abc"), Bencode::Number(2));
        assert_eq!(try_bencode(Bencode::Dict(m)), bytes("d3:abci2e3:abdi3e1:zi1ee"));
    }

    fn assert_decoded_eq(events: &[BencodeEvent], expected: Result<Bencode, Error>) {
        let mut parser = Parser::new(events.to_vec().into_iter());
        let result = parser.parse();
        assert_eq!(expected, result);
    }

    #[test]
    fn decodes_empty_input() {
        assert_decoded_eq(&[], Ok(Bencode::Empty));
    }

    #[test]
    fn decodes_number() {
        assert_decoded_eq(&[NumberValue(25)], Ok(Bencode::Number(25)));
    }

    #[test]
    fn decodes_bytestring() {
        assert_decoded_eq(&[ByteStringValue(bytes("foo"))], Ok(Bencode::ByteString(bytes("foo"))));
    }

    #[test]
    fn decodes_empty_list() {
        assert_decoded_eq(&[ListStart, ListEnd], Ok(Bencode::List(vec![])));
    }

    #[test]
    fn decodes_list_with_elements() {
        assert_decoded_eq(&[ListStart,
                            NumberValue(1),
                            ListEnd], Ok(Bencode::List(vec![Bencode::Number(1)])));
        assert_decoded_eq(&[ListStart,
                            ByteStringValue(bytes("str")),
                            NumberValue(11),
                            ListEnd], Ok(Bencode::List(vec![Bencode::ByteString(bytes("str")),
                                               Bencode::Number(11)])));
    }

    #[test]
    fn decodes_nested_list() {
        assert_decoded_eq(&[ListStart,
                            ListStart,
                            NumberValue(13),
                            ListEnd,
                            ByteStringValue(bytes("rust")),
                            ListEnd],
                            Ok(Bencode::List(vec![Bencode::List(vec![Bencode::Number(13)]),
                                      Bencode::ByteString(bytes("rust"))])));
    }

    #[test]
    fn decodes_empty_dict() {
        assert_decoded_eq(&[DictStart, DictEnd], Ok(Bencode::Dict(BTreeMap::new())));
    }

    #[test]
    fn decodes_dict_with_value() {
        let mut map = BTreeMap::new();
        map.insert(util::ByteString::from_str("foo"), Bencode::ByteString(bytes("rust")));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("foo")),
                            ByteStringValue(bytes("rust")),
                            DictEnd], Ok(Bencode::Dict(map)));
    }

    #[test]
    fn decodes_dict_with_values() {
        let mut map = BTreeMap::new();
        map.insert(util::ByteString::from_str("num"), Bencode::Number(9));
        map.insert(util::ByteString::from_str("str"), Bencode::ByteString(bytes("abc")));
        map.insert(util::ByteString::from_str("list"), Bencode::List(vec![Bencode::Number(99)]));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("num")),
                            NumberValue(9),
                            DictKey(bytes("str")),
                            ByteStringValue(bytes("abc")),
                            DictKey(bytes("list")),
                            ListStart,
                            NumberValue(99),
                            ListEnd,
                            DictEnd], Ok(Bencode::Dict(map)));
    }

    #[test]
    fn decodes_nested_dict() {
        let mut inner = BTreeMap::new();
        inner.insert(util::ByteString::from_str("inner"), Bencode::Number(2));
        let mut outer = BTreeMap::new();
        outer.insert(util::ByteString::from_str("dict"), Bencode::Dict(inner));
        outer.insert(util::ByteString::from_str("outer"), Bencode::Number(1));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("outer")),
                            NumberValue(1),
                            DictKey(bytes("dict")),
                            DictStart,
                            DictKey(bytes("inner")),
                            NumberValue(2),
                            DictEnd,
                            DictEnd], Ok(Bencode::Dict(outer)));
    }

    #[test]
    fn decode_error_on_parse_error() {
        let err = Error{ pos: 1, msg: "error msg".to_string() };
        let perr = ParseError(err.clone());
        assert_decoded_eq(&[perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[NumberValue(1), perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[ListStart,
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[ListStart,
                           ByteStringValue(bytes("foo")),
                           perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[DictStart,
                            perr.clone()], Err(err.clone()));
        assert_decoded_eq(&[DictStart,
                            DictKey(bytes("foo")),
                           perr.clone()], Err(err.clone()));
    }
}

#[cfg(all(test, feature = "nightly"))]
mod bench {
    extern crate test;

    use self::test::{Bencher, black_box};

    use serde::{Serialize, Deserialize};

    use streaming::StreamingParser;
    use super::{Encoder, Decoder, Parser, DecoderResult, encode};

    #[bench]
    fn encode_large_vec_of_usize(bh: &mut Bencher) {
        let v: Vec<u32> = (0u32..100).collect();
        bh.iter(|| {
            let mut w = Vec::with_capacity(v.len() * 10);
            {
                let mut enc = Encoder::new(&mut w);
                let _ = v.serialize(&mut enc);
            }
            black_box(w)
        });
        bh.bytes = v.len() as u64 * 4;
    }


    #[bench]
    fn decode_large_vec_of_usize(bh: &mut Bencher) {
        let v: Vec<u32> = (0u32..100).collect();
        let b = encode(&v).unwrap();
        bh.iter(|| {
            let streaming_parser = StreamingParser::new(b.clone().into_iter());
            let mut parser = Parser::new(streaming_parser);
            let bencode = parser.parse().unwrap();
            let mut decoder = Decoder::new(&bencode);
            let result: DecoderResult<Vec<usize>> = Deserialize::deserialize(&mut decoder);
            result
        });
        bh.bytes = b.len() as u64 * 4;
    }
}
