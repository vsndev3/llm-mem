use crate::ingest::document_tree::{DocumentMeta, DocumentNode, ValueNode};

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub format: &'static str,
    pub mime_type: &'static str,
    pub width: u32,
    pub height: u32,
    pub byte_size: u64,
}

pub fn parse_image_bytes(data: &[u8], byte_size: u64) -> Result<(DocumentNode, DocumentMeta), String> {
    let info = detect_image_info(data, byte_size)?;

    let properties = vec![
        ("format".to_string(), ValueNode::Scalar(info.format.to_string())),
        ("width".to_string(), ValueNode::Scalar(info.width.to_string())),
        ("height".to_string(), ValueNode::Scalar(info.height.to_string())),
        ("mime_type".to_string(), ValueNode::Scalar(info.mime_type.to_string())),
    ];

    let description = format!(
        "{} image, {}x{} pixels, {} bytes",
        info.format, info.width, info.height, byte_size
    );

    let children = vec![
        DocumentNode::Paragraph {
            text: description,
            id: None,
        },
        DocumentNode::KeyValue {
            key: "metadata".into(),
            value: ValueNode::Object(properties),
            id: None,
        },
    ];

    let meta = DocumentMeta::new(info.format, info.mime_type, byte_size);
    Ok((DocumentNode::Document {
        children,
        meta: meta.clone(),
    }, meta))
}

fn detect_image_info(data: &[u8], byte_size: u64) -> Result<ImageInfo, String> {
    if data.len() < 12 {
        return Err("Image data too short (need at least 12 bytes)".into());
    }

    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        let width = u32::from(u16::from_be_bytes([data[16], data[17]]));
        let height = u32::from(u16::from_be_bytes([data[18], data[19]]));
        return Ok(ImageInfo {
            format: "png",
            mime_type: "image/png",
            width,
            height,
            byte_size,
        });
    }

    if data.starts_with(b"\xff\xd8\xff") {
        let mut pos = 2usize;
        while pos + 4 <= data.len() {
            if data[pos] != 0xff {
                break;
            }
            let marker = data[pos + 1];
            if marker == 0xda || marker == 0xd9 {
                break;
            }
            if pos + 4 > data.len() {
                break;
            }
            let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            if (0xc0..=0xc2).contains(&marker) && pos + 9 <= data.len() {
                let height = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as u32;
                let width = u16::from_be_bytes([data[pos + 7], data[pos + 8]]) as u32;
                return Ok(ImageInfo {
                    format: "jpeg",
                    mime_type: "image/jpeg",
                    width,
                    height,
                    byte_size,
                });
            }
            pos += 2 + len;
        }
        return Err("Could not find JPEG dimensions".into());
    }

    if data.starts_with(b"GIF8") {
        let width = u16::from_le_bytes([data[6], data[7]]) as u32;
        let height = u16::from_le_bytes([data[8], data[9]]) as u32;
        return Ok(ImageInfo {
            format: "gif",
            mime_type: "image/gif",
            width,
            height,
            byte_size,
        });
    }

    if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        if data.len() > 30 && &data[12..16] == b"VP8X" {
            let w = u32::from_le_bytes([data[24], data[25], data[26], 0]) & 0xffffff;
            let h = u32::from_le_bytes([data[27], data[28], data[29], 0]) & 0xffffff;
            return Ok(ImageInfo {
                format: "webp",
                mime_type: "image/webp",
                width: w + 1,
                height: h + 1,
                byte_size,
            });
        }
        if data.len() > 30 && &data[12..16] == b"VP8L" {
            let bits = u32::from_le_bytes([data[21], data[22], data[23], data[24]]);
            let width = (bits & 0x3fff) + 1;
            let height = ((bits >> 14) & 0x3fff) + 1;
            return Ok(ImageInfo {
                format: "webp",
                mime_type: "image/webp",
                width,
                height,
                byte_size,
            });
        }
        if data.len() > 26 && &data[12..16] == b"VP8 " {
            let frame = &data[23..26];
            let width = u16::from_le_bytes([frame[0] & 0x3f, frame[1]]) as u32;
            let height = u16::from_le_bytes([frame[1] & 0x0f, frame[2]]) as u32;
            return Ok(ImageInfo {
                format: "webp",
                mime_type: "image/webp",
                width,
                height,
                byte_size,
            });
        }
        return Err("Unsupported WebP variant".into());
    }

    Err("Unsupported image format".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_png(w: u16, h: u16) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"\x89PNG\r\n\x1a\n");

        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

        let len = (ihdr.len() as u32).to_be_bytes();

        data.extend_from_slice(&len);
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr);
        data.extend_from_slice(&0u32.to_be_bytes());

        data.extend_from_slice(b"\x00\x00\x00\x00IEND\xae\x42\x60\x82");
        data
    }

    #[test]
    fn test_png_dimensions() {
        let data = make_png(100, 200);
        let (doc, meta) = parse_image_bytes(&data, data.len() as u64).unwrap();
        assert_eq!(meta.format, "png");
        if let DocumentNode::Document { children, .. } = doc {
            assert!(children.iter().any(|c| matches!(c, DocumentNode::Paragraph { text, .. } if text.contains("100x200"))));
        }
    }

    #[test]
    fn test_jpeg_detected() {
        let jpeg = vec![
            0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x50, 0x00, 0x40, 0x00,
        ];
        let info = detect_image_info(&jpeg, jpeg.len() as u64);
        assert!(info.is_ok(), "JPEG failed: {:?}", info.err());
    }

    #[test]
    fn test_gif_detected() {
        let mut gif = vec![b'G', b'I', b'F', b'8', b'9', b'a'];
        gif.extend_from_slice(&80u16.to_le_bytes());
        gif.extend_from_slice(&60u16.to_le_bytes());
        gif.extend_from_slice(&[0xf0, 0x00, 0x00]);
        let result = parse_image_bytes(&gif, gif.len() as u64);
        assert!(result.is_ok(), "GIF should parse: {:?}", result.err());
    }

    #[test]
    fn test_unknown_format() {
        let data = b"not an image";
        assert!(parse_image_bytes(data, data.len() as u64).is_err());
    }
}
