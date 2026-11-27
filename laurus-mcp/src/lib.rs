//! `laurus-mcp` provides a Model Context Protocol (MCP) server for the laurus
//! search engine.
//!
//! This crate acts as a gRPC client to a running laurus-server instance,
//! exposing its capabilities as MCP tools so that AI assistants such as Claude
//! can index documents and perform searches through the standard MCP stdio
//! transport.
//!
//! # Tools
//!
//! | Tool | Description |
//! |------|-------------|
//! | `connect` | Connect to a running laurus-server gRPC endpoint |
//! | `create_index` | Create a new search index from a JSON schema |
//! | `get_index` | Get index statistics |
//! | `add_document` | Add or upsert a document |
//! | `get_document` | Retrieve a document by ID |
//! | `delete_document` | Delete a document by ID |
//! | `commit` | Commit pending changes to disk |
//! | `search` | Perform lexical / hybrid search |
//!
//! # Usage
//!
//! Typically started via `laurus mcp [--endpoint <url>]` from the `laurus-cli`
//! crate.  The server reads from stdin and writes to stdout (MCP stdio
//! transport).

pub mod convert;
pub mod server;
