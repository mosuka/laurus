# Laurus PostgreSQL Extension: Development Plan

## 0. About this document

- **Purpose**: A single plan for an extension that makes Laurus usable as a PostgreSQL index access method (crate name `laurus-postgresql`). It covers the design of the extension, the order of development, the changes needed in core (`LexicalStore` / `VectorStore`), the measurements and reference implementations the plan rests on, and the later stages.
- **Summary**:
  - Start with stage 1: a `Storage` implementation that keeps Laurus files in PostgreSQL tables.
  - Only four core changes come first (S1, S2, S4, S5 in chapter 3). The extension is then built in the order M1 to M7 (chapter 11).
  - The Catalog (chapter 5) and the format changes (chapter 6) are done after the extension runs and has been measured, and only the items that turn out to be needed.
  - Stage 1 is expected to be practical for read-mostly lexical indexes of a few hundred thousand to a few million documents, where the number of connections can be kept small. Tables with many concurrent writers, deployments with many connections, and large vector indexes need stage 2 or later.
  - Every desk review so far has found a problem that affected the design. The rest will be found by running the items in 2.11 and 7.1 on a prototype.
- **Target**: PostgreSQL 17 or later (`aminsertcleanup`, which reports the end of a statement, was added in 17). Implemented with pgrx.
- **Paths**: Laurus code paths are relative to `laurus/src/`.
- **Names**: The extension, the access method and the schema are all named `laurus` for now; this is revisited if a problem shows up. Operator and function names are placeholders (2.16).
- **Tracking**: #1640.

### How to read

| Goal | Where to read |
| :--- | :--- |
| What to build now | 1.5 (scope) → chapter 3 (core changes) → chapter 2 (extension design) → chapter 11 (order) |
| Why this approach | Chapter 1, chapter 4 (measurements and limits), chapter 10 (reference implementations) |
| What comes later | Chapter 5 (Catalog), chapter 6 (performance and format), chapter 8 (placing data directly in pages) |
| What is uncertain | Chapter 12 |

### Code that was read

| Target | Commit | Use |
| :--- | :--- | :--- |
| `mosuka/laurus` main | `7236416` (#1633; `Cargo.toml` says 0.13.2 but it is 268 commits ahead of the v0.13.2 tag) | The code to change |
| `paradedb/paradedb` (pg_search) | `42bc7bd` | BM25 index that stores Tantivy segments in pages |
| `timescale/pg_textsearch` | `01d637c` | BM25 index written in C. `ARCHITECTURE.md` describes its concurrency and reclamation design |
| `pgvector/pgvector` | `568fea3` | Mutable HNSW on pages |
| `postgres/postgres` | `70c4a57` | The index AM contract (`doc/src/sgml/indexam.sgml`), `generic_xlog.h`, `commands/vacuum.c`, `storage/lmgr/proc.c` |
| `citusdata/citus` (columnar) | `08479ac` | Precedent for an access method that keeps metadata in auxiliary tables |
| `tensorchord/VectorChord` | `0768657` | Vector index that abstracts pages behind traits |
| `timescale/pgvectorscale` | `3ae6fcc` | DiskANN. Switches node layout behind a trait |
| `tensorchord/pgvecto.rs` | `2b290b3` | Own files plus a background worker (superseded by VectorChord) |
| `lanterndata/lantern` | `7df590c` | Stores an external library's (usearch) index in pages |
| `pgroonga/pgroonga` | `3aff122` | Full-text search using Groonga. Only the file layout was checked |

pg_search is AGPL-3.0. It is used as a design reference only; no code is reused.

### Levels of confidence

- **Confirmed**: Checked in the code on main, or in the code and documents of a reference implementation.
- **Proposal**: A design introduced by this document. Trait and function names do not exist in the code yet.
- **Unverified**: Collected in chapter 12. In particular, none of the index AMs that were read uses the stage 1 approach (writing auxiliary tables from inside an index AM). The closest precedent is Citus columnar, a table AM (10.2).

---

## 1. Overview

### 1.1 Three stages

| Stage | Content | What it gives | Limits |
| :--- | :--- | :--- | :--- |
| **1. Table-backed Storage** | Laurus files are rows of auxiliary tables. The doc_id → ctid mapping and deletion marks live in the index's own pages. Writes are serialized by a per-index write lock | A working extension. Transactions, consistent reads, reclamation, WAL, replication and standby are left to PostgreSQL | One writing transaction per index at a time. Reader structures are loaded into memory per backend |
| **2. Catalog** | The segment list becomes a table with one row per segment, and core supports multiple writers | Concurrent writes. Reader caches survive commits | Many core changes (chapter 5) |
| **3. Performance** | Absorbing small writes, range reads, reader memory, format changes. Direct placement in pages if needed | Performance at practical scale | Includes breaking format changes |

### 1.2 Why this order

- **The weak points of stage 1 can be found by building and measuring.** I/O overhead, memory and the handling of auxiliary tables can be measured on a prototype. Performance down to the storage layer has already been measured (4.1). The hard parts of placing data directly in pages (page reclamation, safety on a standby) cannot be shown correct without building them out.
- **Little is thrown away.** The thread feature gate, the doc_id → ctid array, and the search and planner integration are the same in every stage.
- **Tables stay useful in stage 2.** With a table-backed Catalog, publishing is a row INSERT, a merge is DELETE plus INSERT, and consistency is guaranteed by MVCC. There is no need to write a metapage and linked lists by hand.
- **Direct placement in pages is the last option.** pg_search offers reads on a standby only in its Enterprise edition; the public edition refuses them. That is how hard this part is. It is considered when measurements show I/O to be a problem.

### 1.3 Uses this does not fit

If the main target is tables with many concurrent writers (OLTP) from the start, stage 1 is a detour. Do stage 2 first in that case.

### 1.4 Order relative to other initiatives

Four initiatives are being considered in parallel: the PostgreSQL extension, a columnar doc store, object storage, and SQL support.

**Dependencies**

| Initiative | Depends on |
| :--- | :--- |
| PostgreSQL extension (stage 1) | Nothing. It does not use the doc store (the source of truth is the heap) or the SQL layer (PostgreSQL handles SQL). The only core changes needed are the four in chapter 3 |
| Columnar doc store | Assumes its file format is decided together with object storage (Bundle) |
| Object storage | Needs coordination of multiple writers (Catalog), range reads, lazy posting decode, and structures readable in place. The requirements overlap with stages 2 and 3 of the PostgreSQL extension |
| SQL support | Queries that do not use an index read the column store, so it depends on the columnar doc store |

**Recommended order**

1. Shared groundwork (S1, S2, S4, S5 in chapter 3).
2. Stage 1 of the PostgreSQL extension (lexical prototype and measurement).
3. Decide the formats together. Design the object storage (Bundle) and column store formats in one pass, using the stage 1 measurements and the format changes in chapter 6 (F1 to F4).
4. Implement object storage and the column store. The Catalog is built as one trait for both Bundle and stage 2 of the PostgreSQL extension (5.3).
5. SQL support.
6. Stage 2 and later of the PostgreSQL extension (the items that measurements show are needed).

**Reasons**

- **The format breaks only once.** The candidate format changes in chapter 6 (segment-relative doc_id, lazy posting decode, an offset table for the HNSW graph, in-place dictionary reads) touch the same places as the Bundle and column store formats. With the PostgreSQL measurements done first, they can be built into the Bundle design from the beginning. In the reverse order, the format changes again after Bundle is fixed.
- **Stage 1 does not depend on the format.** A table-backed Storage works at file granularity, so the extension keeps working almost unchanged when the format changes later.
- **Measurements make the other designs concrete.** In the measurements of 4.1, what is slow right after opening is not I/O but building the term dictionary, and memory is proportional to vocabulary × number of segments. The same applies to a cold start on object storage.
- **SQL is last because it has the most dependencies.** Without the column store it cannot serve queries that do not use an index. Once the PostgreSQL extension works, some users who want SQL can be served there first.

**Points to decide before the formats (step 3)**

- **How doc_ids are held**: Stage 1 assumes a doc_id that is unique across the index (the doc_id → ctid array and deletion marks; 2.1). If doc_ids become segment-relative (F3 in chapter 6), that array no longer works. Decide whether to keep "a logical id unique across the index" separate from "a position inside a segment". This also relates to decisions made for the column store ("the `_id` → doc_id mapping lives in the column store", "segment boundaries are independent of the index").
- **Representation of pushdown**: A PostgreSQL Custom Scan and a DataFusion table function answer the same question: which filters, top-k and aggregates can be handed to Laurus. Put one shared representation in core and have both use it.

**When the opposite decision applies**

- If the main use of the PostgreSQL extension is, from the start, tables with many concurrent writers or large vector indexes, stage 1 is not enough. The Catalog and the format changes become prerequisites, so move object storage forward first and build the shared parts.
- A DuckDB extension fits the column store well (analytic queries read columns), so it comes after step 4.

### 1.5 Scope of stage 1

**In scope**

- An index that can be created with `CREATE INDEX ... USING laurus`.
- Lexical (BM25) search. Vector search is added on the same structure after lexical works.
- INSERT / UPDATE / DELETE with transactions, VACUUM, crash recovery, and streaming replication.

**Out of scope**

| Item | Reason |
| :--- | :--- |
| Concurrent writes | Serialized by a per-index write lock. Removing it needs Catalog support in core (chapter 5) |
| Indexes on unlogged / temporary tables | After a crash, the index pages and the auxiliary tables disagree. Rejected at creation (2.11) |
| Parallel scan, parallel build | Each backend holds its own store, so memory multiplies. There is one writer |
| Embedding generation | Precomputed vectors are accepted |
| Unified scores under distribution (Citus and similar) | Statistics differ per shard. Considered separately |

---

## 2. Stage 1 design: table-backed Storage (proposal)

### 2.1 Structure

```text
SQL: operators / functions / (later, Custom Scan)
laurus-postgresql (pgrx)
  ├ am        Index AM callbacks (2.13)
  ├ pages     The index's own pages: metapage, doc_id → ctid array (2.12)
  ├ storage   TableStorage (implementation of Laurus's Storage trait; 2.3)
  ├ catalog   Auxiliary table definitions and access, storage id management (2.2, 2.12)
  ├ lock      Per-index write lock (2.5)
  ├ session   Backend-local state and transaction callbacks (2.12)
  ├ worker    Background worker: merge, delete folding, sweep (2.14)
  └ sql       Operators, functions, GUCs, index options (2.16)
laurus core
  ├ LexicalStore (almost unchanged)
  └ VectorStore (almost unchanged)
PostgreSQL
  ├ The index's own pages: doc_id counter, doc_id → ctid array, deletion marks
  │   (not transactional; pages are updated directly and WAL-logged, like an ordinary index)
  └ Auxiliary tables: chunks of Laurus files (MVCC, WAL, VACUUM, replication)
```

**Why the data is split in two places**:

- Laurus files (segments and manifests) should become durable together with the transaction, so they go in auxiliary tables.
- Deletion marks are written by VACUUM. VACUUM cannot perform ordinary updates on tables (2.8), so the marks go in the index's own pages. The doc_id → ctid mapping is kept in the same place so that VACUUM can walk it without reading a table.
- Citus columnar splits its data the same way (bulk data in the relation's own pages, metadata in tables; 10.2).

### 2.2 Auxiliary tables

- The extension owns three shared tables (DDL in 2.12): the mapping from index to storage (`laurus.storage`), the file list (`laurus.file`), and the file contents (`laurus.chunk`, primary key `(storage_id, file_id, chunk_no)`). `storage_id` is issued from a sequence and written to the index metapage (the same as Citus columnar; the relfilenode itself is not used). When `REINDEX` or `TRUNCATE` recreates the index storage, the new metapage gets a new id.
- Tables are not created per index, to match the structure of Citus columnar (shared tables such as `columnar.stripe`, looked up by storage id; `columnar_metadata.c`). Tables owned by an extension are not dumped by `pg_dump`, and no table DDL runs when an index is created or dropped. Rows of an old storage are deleted by storage id (the same as `DeleteMetadataRows` in Citus columnar). A table AM has a callback that reports a storage switch (`relation_set_new_filelocator`); an index AM does not. So the mapping between index OID and storage id is kept in `laurus.storage`, and `ambuild` / `ambuildempty` delete "rows of the same index other than the new id". `DROP INDEX` is caught with an object access hook or an event trigger. Both run inside the transaction, so if the DDL rolls back, the deletion rolls back too. To cover anything missed, the background worker has a sweep that deletes rows of indexes that no longer exist.
- The content key is an integer so that a file name is not stored per chunk. A rename is an UPDATE of one row in the file list.
- Chunks are either small enough not to go to TOAST (under about 2 KB), or the storage of `data` is set to `EXTERNAL` to turn off compression. Laurus files are already compressed (bit-packing for postings, LZ4 for stored fields).
- Built-in large objects (`lo_*`) also support seek and partial reads, but they all live in one catalog for the whole database, so they are not used.
- Privileges on the auxiliary tables are revoked from PUBLIC. If the index contents (terms and postings) were readable through SQL, users could infer content while bypassing row-level security or column privileges. Reads and writes from inside the access method use low-level functions, so users need no privileges.
- Autovacuum settings are given per auxiliary table. Old manifest rows become dead on every commit and old segment rows on every merge, so the default thresholds lead to bloat.

### 2.3 Mapping to the Storage trait

`Storage` is a file-name-based trait with 15 methods (`storage.rs`). `LexicalStore` and `VectorStore` do not take the storage lock (only `Engine` does; `_storage_lock` in `engine.rs`).

| Storage operation | Implementation in TableStorage | Note |
| :--- | :--- | :--- |
| `create_output` | Buffer in memory; INSERT chunk rows on `close` | Segment files do not change after close |
| `create_output_append` | Add rows after the existing last chunk | Used on the lexical / vector side by the deletion log (`maintenance/deletion.rs`). The `Engine` WAL is not used |
| `open_input` | `Read + Seek` implemented with chunk range reads. `as_slice` returns `None` | The copy path for storages without `as_slice` already exists and has tests (`NoSlice`) |
| `rename_file` | UPDATE of `file_name` (DELETE the target first if it exists) | Atomic, because it is in the same transaction |
| `delete_file` | DELETE of rows | Space is reclaimed by VACUUM. Readers on an old snapshot can keep reading |
| `list_files` / `file_exists` / `file_size` / `metadata` | The file list is held in memory per snapshot | `file_exists` is called from about 100 places in lexical and vector, so it must not hit the table each time |
| `sync` | Nothing | Durability comes from PostgreSQL's WAL |
| `lock_manager` | `None` | The write lock is taken by the extension (2.5) |

**Implementation rules**

| Point | Content |
| :--- | :--- |
| How to read and write | Not through SQL. Use executor helper functions and index scans (see below) |
| Batching reads | Fetch consecutive chunks with one index range scan instead of one lookup per chunk. Keep relations and scans open for the duration of a query |
| Chunk cache | Cache chunks per backend. Segment files are immutable, so no invalidation is needed |
| Snapshot | A writer uses the latest snapshot taken after acquiring the lock (2.6). A reader uses the query snapshot (2.7). They are not mixed within one scan |
| Interrupts | Call `CHECK_FOR_INTERRUPTS` on every read and write (2.11) |
| Errors | Discard the store after an error (2.11) |
| Threads | Build Laurus without threads (S1). Auxiliary tables can only be touched from the backend's main thread |
| Readers are read-only | In a store that does not hold the write lock, `delete_file` in TableStorage does nothing (actual deletion is left to the background worker's sweep). `create_output` and `rename_file` return an error (a reader trying to write is unexpected, so it is detected, not silently dropped). Laurus runs an orphan sweep when it opens a store and tries to delete files that are not in the manifest (`sweep_orphans` in lexical, `cleanup_orphans` in vector). If a reader ran it, a search would write to tables, which fails on a standby and in read-only transactions |
| Chunk size | Start at about 2 KB and decide by measurement. In measurements, one search read 60 to 70 chunks (4.1) |

**Reading your own writes within the same statement**:

- In PostgreSQL, rows written within a command are not visible to an ordinary snapshot until the command counter advances. Laurus sometimes reads back a file it has just written (segment, manifest).
- Citus columnar calls `CommandCounterIncrement()` after writing metadata rows and reads with `GetTransactionSnapshot()` (`columnar_metadata.c`). Either do the same, or keep the files written in the transaction in backend memory and overlay them on reads. The snapshot a writer uses is described in 2.6.
- Rows can be written with executor helper functions (`ExecSimpleRelationInsert`, `simple_heap_delete`), as Citus columnar does. SQL is not needed.

### 2.4 What is left to PostgreSQL

| Item | Mechanism |
| :--- | :--- |
| Transactions | Written rows are invisible to other backends until commit and disappear on abort |
| Consistent reads | Each backend sees the set of files through its own snapshot. The manifest and the segments never disagree |
| Safety of the orphan sweep | Uncommitted files of other backends do not appear in `list_files`, so the sweep at open does not delete them |
| File reclamation | Deleted rows are reclaimed by VACUUM once no snapshot can see them |
| Crash recovery, PITR, replication, standby | The heap WAL applies as is |

The hard parts of placing data directly in pages (chapter 8), namely page reclamation, conflict records for a standby, and the order of WAL records, do not have to be handled in this approach. The array kept in the index's own pages (2.1) is only appended to and has flags updated in place; its pages are never reclaimed or reused.

### 2.5 Serializing writes

**Why it is needed** (confirmed):

- Lexical `publish_with` (`inverted/segment_manifest.rs`) applies a delta to the in-memory segment list and saves it. `InvertedIndex::refresh` does not re-read storage (#1023).
- Vector `SegmentManager` (`vector/index/segment/manager.rs`) mutates the in-memory list first and then saves the whole list.
- Both save without knowing about segments another process published earlier, and so remove them from the manifest.

**Method**:

- Take a per-index write lock on the first write and hold it until the end of the transaction. Another writer's uncommitted manifest is invisible, so the lock cannot be released at the end of the statement.
- The lock is an ordinary lock on a dedicated object (`LockDatabaseObject` or similar). Advisory locks can collide with keys used by applications.
- What is serialized: INSERTs into the table that has the index, and UPDATEs that change an indexed column (HOT updates do not go through the index and are not affected). The span from the first write to the transaction commit is serialized per transaction.
- After taking the lock, re-read the manifest (S5). This guarantees seeing the segments of the writer that committed just before.
- Merges run under the same lock.
- VACUUM does not take the lock. Deletion marks are written to the index's own pages and Laurus files are not touched (2.8). Deletion marks are keyed by doc_id and independent of segments, so the race between merge and deletion (C9 in chapter 5) does not occur in stage 1.

**Notes**:

- Deadlocks increase. Besides writing to several indexes in different orders, they also occur as "A holds the index lock and waits for B's row lock, while B waits for A in a write to that index". PostgreSQL detects them and raises an error in one transaction.
- Merges also run under the lock, so writes wait in the meantime. Merging large segments takes long (the lexical merge rebuilds by replaying postings). Stage 1 accepts the wait.
- Autovacuum must not hold the lock. PostgreSQL cancels an autovacuum worker that is blocking another backend's lock wait (`proc.c`). With a design in which VACUUM takes the write lock, autovacuum would never finish while writes continue.

### 2.6 Write flow

1. `aminsert`: take the write lock, re-read the manifest (S5), allocate a doc_id, and add the document to the writer's buffer.
   - The doc_id comes from a counter in the index metapage. Like a sequence, it is not transactional and does not go back on abort. The store layer has no counter (numbering is the job of `DocumentLog` in `Engine`).
   - It must not be derived from the largest doc_id in the manifest. When the last documents are deleted and removed by a merge, the maximum goes down and the same doc_id is issued again. The vector deletion bitmap is keyed by doc_id, so the reissued document would be hidden as deleted.
   - Write the row's ctid at the position of the allocated doc_id (the doc_id → ctid array). After an abort the array entry remains, but no segment references it, so it never appears in search.
2. End of statement (`aminsertcleanup`, PG17 or later): call Laurus commit. Segments and the manifest are written to the auxiliary tables. Advance `version` in `laurus.storage` (2.12). The rows are still uncommitted, so other backends do not see them. Later statements in the same transaction do.
3. Transaction commit: the auxiliary table rows become durable and the lock is released.
4. Abort: the rows disappear. Discard the in-memory store.
5. Savepoint rollback: only the rows written inside it disappear. The in-memory store is now ahead of the rows, so discard the writer and re-read the manifest.

pg_search delays the commit until `PRE_COMMIT` in the logical replication apply worker. That is an alternative when writing at the end of the statement is not possible.

**The snapshot a writer uses**: After taking the write lock, a writer reads the manifest and files with the latest snapshot (`GetLatestSnapshot`). It must not read with the statement or transaction snapshot. Even under READ COMMITTED the snapshot is taken at the start of the statement and the lock wait comes after it, so a manifest committed by another writer during the wait is invisible. Under REPEATABLE READ and above the snapshot is fixed at the start of the transaction, so even more is invisible. Overwriting based on an old manifest loses that writer's segments. Because the lock is held, "the latest committed state plus my own writes" is the correct state. A reader reads with the query snapshot (2.7).

**Target PostgreSQL**: `aminsertcleanup` exists in PostgreSQL 17 and later. Earlier versions need another way to catch the end of a statement (an executor hook or `PRE_COMMIT`). Stage 1 targets 17 and later.

### 2.7 Read flow

1. At the start of a scan, read `version` in `laurus.storage` (2.12) with the query snapshot.
2. If it differs from the version of the backend-local store, re-read the manifest (S5). Readers of unchanged segments are reused.
3. Search, and map doc_ids to ctids through the array. Drop entries that carry a deletion mark. Return the rest; visibility is decided by the heap.

**Why following is needed**: Under READ COMMITTED the snapshot changes per statement. A store at an old version references files that a later merge removed, and those rows are invisible to the new snapshot.

**Why the store is not reopened** (measured, 4.1): Reopening the store re-reads the term dictionaries and other structures of every segment. That takes 0.16 to 2.0 seconds on a 200k-document index, and every reader pays it on every foreign commit. It does not work under a mixed read and write load.

### 2.8 Deletion (VACUUM)

**Constraint** (confirmed in PostgreSQL's `vacuum.c`):

- Lazy VACUUM runs with `PROC_IN_VACUUM` set. Other VACUUMs ignore this backend when they decide the reclamation horizon.
- So no ordinary updates are allowed inside lazy VACUUM (comment on `vac_update_relstats`: another VACUUM may decide that tuples written with our xid can be removed).
- For the same reason, tuples of other tables that the lazy VACUUM snapshot needs may be reclaimed (comment on `vacuum_rel`).
- Therefore `ambulkdelete` must not write to the auxiliary tables and cannot rely on reading them.

**Design**:

- `ambulkdelete` walks the doc_id → ctid array in the index's own pages and sets a deletion mark on every entry the callback reports as dead. This is a direct page update with WAL, the same as an ordinary index AM.
- Laurus files are not touched. Search checks the mark of each hit doc_id and drops marked hits (2.7).
- Reuse of a ctid is safe. The mark is on the doc_id, and a new row has a new doc_id.
- Parallel VACUUM does not need to be disabled (nothing is written to the auxiliary tables, so no xid is assigned).

**Delete folding** (required for tables that have updates or deletes):

- Marked documents stay in Laurus segments, keep occupying top results, and stay in the BM25 statistics.
- The background worker, in an ordinary transaction and under the write lock, passes marked doc_ids to the Laurus delete API (the `LexicalStore` delete for lexical, `SegmentedCore::mark_deleted` for vector). Later merges remove the data.
- The marks in the array are not cleared. Correctness rests on the array marks alone. Without folding, Laurus does not know about the deletions, so merges do not remove the data and the index does not shrink. Marked documents stay at the top and re-searches (2.9) increase.
- Whether an entry has been folded is kept as a second flag on the array entry. Even if the folding transaction aborts, search only looks at the first flag (dead), so results stay correct.
- `CREATE INDEX CONCURRENTLY` validation also uses `ambulkdelete` (`validate_index` collects every TID in the index with a callback that deletes nothing). Walking the array and calling the callback on every entry covers it.

**Comparison with a reference implementation**: pg_search keeps the ctid in a fast field of each segment and records deletions by walking each segment in `ambulkdelete`. pg_search keeps all its data in the index's own pages, so it does not hit the VACUUM constraint.

### 2.9 Consistency with the heap

- The index contents become durable together with the transaction, so entries of aborted writes do not remain in the index.
- DELETE and UPDATE are not reported to the index. Entries of old rows remain until VACUUM and are rejected by the heap visibility check.
- After the top k are taken, some are dropped by deletion marks and heap visibility. If too few remain, search again with a larger limit. With folding (2.8) running, the share that is dropped stays small.
- Search returns the top k in one batch and looks at the heap afterwards. According to the PostgreSQL documentation (the section on index locking), this is allowed for MVCC snapshots and not for non-MVCC snapshots.

### 2.10 Vector

- `VectorStore` runs on the same TableStorage and write lock.
- The write API is async (`lock_writer().await`, `commit().await`). With precomputed vectors the only thing awaited is the tokio Mutex, so the extension runs it synchronously.
- Embeddings are produced outside, through the precomputed embedder (`embedding/precomputed.rs`).
- The problems that remain in stage 1 are in 4.2 and chapter 6.
- In a build without threads, HNSW construction is also serial (`hnsw/writer.rs` uses rayon). `CREATE INDEX` on a large table becomes slow. Keeping parallelism only for pure computation that does not touch storage (graph construction) can be considered as part of how S1 splits the features. The time has not been measured.

---

### 2.11 Considerations (added in reviews)

**Transactions and isolation levels**

| Point | Content |
| :--- | :--- |
| Writer snapshot | Regardless of isolation level, read with the latest snapshot taken after acquiring the lock (2.6) |
| SERIALIZABLE | The auxiliary tables are ordinary tables, so they are subject to predicate locks (PostgreSQL exempts only system and temporary relations; `predicate.c`). Readers read manifest rows and writers rewrite them, so transactions that do not really conflict can get serialization failures. Stated as a limit of stage 1 |
| 2PC (`PREPARE TRANSACTION`) | Everything is flushed at the end of the statement, so nothing unwritten remains in backend memory. The lock is carried by the prepared transaction, and other writers wait until it commits |
| Scan and write in the same backend | With a cursor still open, the same backend may write and re-read the manifest. A scan pins its reader at the start |

**DDL and lifecycle**

| Point | Content |
| :--- | :--- |
| `REINDEX`, `TRUNCATE`, `DROP INDEX` | Switch of storage id and deletion of old rows (2.2) |
| `CREATE INDEX CONCURRENTLY` | Works but brings no benefit. The builder holds the write lock during the build, so INSERTs wait |
| Unlogged tables | After a crash the index returns to its initial state at build time (init fork), but the auxiliary table rows remain. Searching in that state returns wrong results. Stage 1 rejects index creation on unlogged and temporary tables |
| Tablespace | A tablespace given to the index does not apply to the auxiliary tables |
| `pg_dump` | The contents of extension-owned tables are not dumped. On restore, the index is rebuilt by `CREATE INDEX` |
| Logical replication | A `FOR ALL TABLES` publication also includes extension-owned tables (`is_publishable_class` in `pg_publication.c` covers every permanent table created after initdb). Changes to the auxiliary tables flow to subscribers, where meaningless rows pile up or, if the tables do not exist, errors occur. Document how to exclude them from a publication (list tables explicitly, or use an exclusion clause where the version supports it) |
| Standby | Both the auxiliary tables and the index pages are replicated through WAL, so search should work on a standby. When the auxiliary tables are vacuumed on the primary, long queries on the standby are cancelled under the usual rules |

**Errors and interrupts**

| Point | Content |
| :--- | :--- |
| PostgreSQL errors | When an error occurs inside TableStorage (lock timeout, cancel, disk full and so on), the stack unwinds from the middle of a Laurus call. pgrx turns the error into a Rust panic to unwind, so Laurus's locks (`std::sync` `RwLock` / `Mutex`) can be left poisoned. After an error, discard and rebuild the stores held by the backend |
| Query cancel | Computation inside Laurus (posting decode, scoring, merge) does not check for interrupts. Call `CHECK_FOR_INTERRUPTS` on every TableStorage read and write. Long computation that does not touch storage cannot be cancelled in the meantime |
| Relcache invalidation | When the index is rebuilt or dropped, discard the store held by the backend |

**Operations**

| Point | Content |
| :--- | :--- |
| Bulk load | One statement creates many segments (flush every 10,000 documents by default). With merging moved out of the statement (S4), the number of segments stays high until the background worker catches up. Allow a setting that merges inside the statement for bulk loads |
| Analyzer dictionaries | The analyzer is used both at index time and when parsing queries, so it is loaded per backend. Morphological dictionaries are large, so place them so that the file can be mmapped and shared between processes |
| Parallel scan | If every worker opens a store, memory multiplies by the number of workers. Not supported at first |
| WAL volume | A segment goes into WAL in full when written and again on every merge. This affects replication lag and archive volume. The volume has not been measured |
| Format version | A release that changes Laurus's on-disk format (the format changes of chapter 6, or the introduction of Bundle) makes existing indexes unreadable. Detect it with the format version in the metapage and raise an error asking for `REINDEX`. `pg_upgrade` carries the index files and the auxiliary tables over as they are, so the same detection is needed |
| Memory management | Memory allocated by Laurus is in neither PostgreSQL's memory contexts nor settings such as `work_mem`. The limit is given through Laurus-side settings (buffer and cache sizes) |

### 2.12 Data structures

**Metapage (block 0 of the index)**

| Field | Content |
| :--- | :--- |
| magic, format version | Identifies the extension version |
| `storage_id` | Key into the auxiliary tables. Issued from a sequence (2.2) |
| `next_doc_id` | The next doc_id to issue. Like a sequence, it does not go back on abort (2.6). Either WAL-log every allocation, or advance it in batches before use as sequences do, so that no number is reissued after a crash |
| Array tail | The last array page in use and the position in it |

**doc_id → ctid array (block 1 and later of the index)**

- Fixed-length entries in doc_id order. The block number and position are computed from the doc_id.
- An entry is 8 bytes: ctid (6 bytes) and flags (2 bytes).
- There are two flags. `dead` is set by VACUUM and read by search. `folded` means the deletion has been passed to Laurus; search does not read it (2.8, 2.14).
- Pages are only appended to and have flags updated in place. They are never reclaimed or reused. Updates use `GenericXLog`.
- 8 bytes per document. Entries are added on every UPDATE and the array does not shrink until `REINDEX`.

**Auxiliary tables**

```sql
-- Mapping from index to storage. version advances whenever Laurus files change
CREATE TABLE laurus.storage (
    storage_id bigint PRIMARY KEY,
    index_oid  oid    NOT NULL,
    version    bigint NOT NULL
);

-- File list
CREATE TABLE laurus.file (
    storage_id bigint NOT NULL,
    file_id    int    NOT NULL,
    name       text   NOT NULL,
    size       bigint NOT NULL,
    PRIMARY KEY (storage_id, file_id),
    UNIQUE (storage_id, name)
);

-- File contents
CREATE TABLE laurus.chunk (
    storage_id bigint NOT NULL,
    file_id    int    NOT NULL,
    chunk_no   int    NOT NULL,
    data       bytea  NOT NULL,
    PRIMARY KEY (storage_id, file_id, chunk_no)
);
```

- `version` tells readers that Laurus files have changed. A reader reads this one row at the start of a scan and re-reads the manifest if the value differs from its own (not greater or smaller; not equal) (2.7).
- `version` is advanced by every transaction that writes Laurus files. That includes not only insert commits but also merges and delete folding by the background worker. The vector deletion bitmap (`.delmap`) is rewritten without going through the manifest, so using only manifest changes as the trigger would miss it.
- The reasons for this layout, the privileges and the autovacuum settings are in 2.2.

**Backend-local state (per index)**

| State | Content |
| :--- | :--- |
| Store | `LexicalStore` (and `VectorStore`) |
| Loaded version | The value of `laurus.storage.version` |
| Writer | Held only while writing. Committed at the end of the statement |
| Write lock held | Until the end of the transaction |

When to discard: transaction abort, savepoint rollback, a PostgreSQL error, relcache invalidation (the index is rebuilt or dropped).

### 2.13 Index AM callbacks

| Callback | Behavior in stage 1 |
| :--- | :--- |
| `ambuild` | Issue a new `storage_id` and create the metapage. Delete the rows of old storages of the same index (2.2). Scan the table, add every row, and commit |
| `ambuildempty` | For unlogged tables. An error in stage 1 |
| `aminsert` | Take the write lock (first call). Re-read the manifest (first call). Allocate a doc_id, write the ctid to the array, and add to the writer's buffer (2.6) |
| `aminsertcleanup` | Call Laurus commit and advance `version` (2.6) |
| `ambulkdelete` | Walk the array and set `dead` on entries the callback reports as dead. Do not touch the auxiliary tables (2.8) |
| `amvacuumcleanup` | Return statistics only. Do not touch the auxiliary tables |
| `ambeginscan` / `amrescan` | Check the version and re-read the manifest if it changed. Parse the query (2.7) |
| `amgettuple` | Search, map doc_ids to ctids through the array, drop `dead` entries, and return |
| `amgetbitmap` | Put the ctids of all matching documents in the bitmap. Laurus needs an entry point that returns the set of matching doc_ids (unverified, chapter 12). Counting queries also take this path. `total_hits` of a search result cannot be used as the match count under top-k collection (chapter 12) |
| `amcostestimate` | A fixed estimate at first |
| `amoptions` | Index options such as the analyzer and field settings |
| `amvalidate` | Validation of operator classes |

### 2.14 Background worker

Runs in ordinary transactions and under the write lock.

| Task | Content |
| :--- | :--- |
| Delete folding | Pass doc_ids that are `dead` and not `folded` to the Laurus delete API and set `folded` (2.8) |
| Merge | Call the Laurus merge (decoupled from commit by S4). It runs under the lock, so writes wait in the meantime (2.5) |
| Sweep | Delete rows whose `storage_id` belongs to an index that no longer exists (2.2) |

### 2.15 DDL

| DDL | Behavior |
| :--- | :--- |
| `CREATE INDEX` | `ambuild` |
| `REINDEX`, `TRUNCATE` | The index storage is recreated and `ambuild` issues a new `storage_id`. The old rows with the same `index_oid` are deleted in the same transaction |
| `DROP INDEX` | Caught with an object access hook or an event trigger; the rows are deleted (which one is reliable is unverified, chapter 12) |
| `CREATE INDEX CONCURRENTLY` | Works but brings no benefit (2.11) |
| DDL rollback | The row deletion rolls back as well |

### 2.16 SQL surface (undecided)

The shape of the operators and functions has not been decided. The first prototype starts with the smallest form.

```sql
CREATE EXTENSION laurus;

CREATE INDEX docs_idx ON docs USING laurus (body);

-- Match
SELECT id FROM docs WHERE body @@@ 'postgres AND search';

-- Ranked
SELECT id FROM docs
WHERE body @@@ 'postgres search'
ORDER BY laurus_score(docs) DESC
LIMIT 10;
```

**To decide**

| Point | Options |
| :--- | :--- |
| Multi-column index | Put text, vector and geo in one index (so that hybrid fusion runs inside Laurus) |
| How queries are passed | A string DSL or a structured type |
| How scores are obtained | A function (`laurus_score`) or an ordering operator. With a function, the index scan returns every matching row and PostgreSQL sorts afterwards (it cannot stop at the top k). With an ordering operator (`ORDER BY body <@> 'query' LIMIT 10`, the form pg_textsearch uses), the index returns rows in score order and `LIMIT` stops early. For a prototype without a Custom Scan, this is the shortest path to top k |
| Vector search | Accept `ORDER BY embedding <=> $1` through an operator class |
| Hybrid | A function that takes the fusion method and weights |
| Top-k pushdown | An index scan at first. Later a Custom Scan that hands `LIMIT`, score order and filters on other columns to Laurus |
| Choosing the analyzer | Index option |
| Names | Operator and function names. The extension, access method and schema are named `laurus` for now |

The representation of pushdown is shared with the DataFusion SQL layer (1.4).

---

## 3. Core changes needed for stage 1

Stage 1 needs four changes: S1, S2, S4 and S5. S5 was added after the measurements (4.1). S3 became unnecessary in a design review (the number is kept as a gap).

### S1. Feature gate for threads

- **Current state** (confirmed): The gates for rayon and related code are on `target_arch = "wasm32"` (91 sites), not on `feature = "native"` (18 sites). `cargo check -p laurus --no-default-features` fails on Linux with 33 errors. The per-segment fan-out (`search_per_segment_fanout` in `inverted/searcher.rs`, `vector/index/segment/fanout.rs`) always uses `par_iter` on native targets when there is more than one segment.
- **Problem**: Auxiliary tables and the buffer manager can only be touched from the backend's main thread.
- **Change**: Split rayon, background threads, mmap and the filesystem into features (for example `threads`, `mmap`, `fs`). The default `native` enables all of them. Add a native build without threads to CI.
- **Basis**: The wasm build is already a configuration without threads or mmap and runs in CI. pg_search also uses a serial writer (`SerialIndexWriter`).
- **Independence**: Depends on nothing else. It can be the first PR.

### S2. Public store APIs

- **Current state** (confirmed): `LexicalStore::{prepare_upsert, apply_prepared_upsert, delete_document_by_internal_id, doc_values_batch}`, `VectorStore::{write_prepared, lock_writer, delete_document_by_internal_id}` and others are `pub(crate)`.
- **Change**: Make the operations needed by callers that do not go through `Engine` `pub`. The doc_id can already be passed from outside (`LexicalStore::upsert_document(internal_id, doc)`). `_id` is a concept of the `Engine` layer and is not brought into the store layer.
- **Test**: The stores work correctly on an index without an `_id` field.

### S3. External key column (no longer needed)

- The original plan kept the ctid as a stored + DocValues field in Laurus.
- After review, the doc_id → ctid mapping lives in the index's own pages (2.1, 2.8). Laurus segments only need the doc_id, so this change is not needed.
- Checked along the way: a document with no stored fields survives a lexical merge (3 segments and 3000 documents merged into one by `optimize`; the number of search hits did not change). An index-only `body` with no other fields can be used.

### S4. A switch to stop auto-merge

- **Current state** (confirmed): The lexical commit runs `maybe_merge` right after the writer's commit (`LexicalStore::finish_commit`). The only way to stop it is to raise `max_segments`. The vector `maybe_auto_compact` runs for the structural limit regardless of the `auto_compaction` setting.
- **Problem**: In stage 1 the commit runs inside the user's transaction. A merge there makes the transaction hold the lock for a long time.
- **Change**: Add an explicit switch and make merge callable on its own. The extension calls merge from the background worker under the write lock.

### S5. Manifest reload (proposal)

- **Current state** (confirmed): `InvertedIndex::refresh` does not re-read storage (#1023). The vector `VectorIndex::refresh` is a no-op by default and searchers are built from the in-memory segment list. The only way to see someone else's commit is to rebuild the store.
- **Problem** (measured, 4.1): Rebuilding the store re-reads everything, including unchanged segments. 0.16 to 2.0 seconds for 200k documents.
- **Change**: Add an operation to the stores that re-reads the manifest from storage and replaces the in-memory list.
  - Lexical: `segment_manifest::load` already exists. Replace the list and drop cached searchers. Readers are built through `segment_reader_cache` (looked up by `SegmentKey`), so readers of unchanged segments are carried over.
  - Vector: Make `SegmentManager::load_state` (private today and called only at construction) callable again. Readers are held by `SegmentedReaderCache` by segment id and are carried over. Re-read the deletion bitmap and advance `bitmap_epoch`.
- **Preconditions**: Only a writer holding the write lock, or a reader, calls it. It is not called while a writer holds unpublished writes in its buffer. With these preconditions, the change is smaller than the stage 2 Catalog (which includes conflict detection and retries).
- **Relation to #1023**: What #1023 stopped was the re-read of `metadata.json` (the path through which an inflated count from the merge engine came back into memory). What is re-read here is `segments.json`, and counts are already derived from the manifest (#1212).
- **Relation to stage 2**: This is the single-writer version of "following by version" (C2) in the Catalog. Stage 2 replaces it.

### Changes not needed in stage 1

| Stage 2 item (chapter 5) | Why stage 1 does not need it |
| :--- | :--- |
| Catalog trait (C1), refresh by version comparison (C2) | Replaced by the write lock and the manifest reload (S5) |
| Segment name numbering (C3) | There is always one writer and the store is opened after the lock, so numbers do not collide |
| Explicit orphan sweep (C4) | Uncommitted files of others are invisible, so they are not deleted by mistake. But opening a store runs the sweep and calls `delete_file` if orphans exist. In a reader's store, TableStorage refuses write operations to prevent this (2.3) |
| Split commit (C5) | The existing `commit()` is called as is (only the S4 switch is needed) |
| Merge and deletion consistency (C9), `.delmap` races (L3) | Serialized by the write lock |
| Multiple writers (L2, V4) | There is one writer |
| `SegmentManager` save-then-swap (V1) | Not needed as a multi-process issue. Useful as a single-process correctness improvement, so it can go in on its own |

---

## 4. Limits of stage 1 and the decision to move to stage 2

### 4.1 Measurements (lexical)

Whether stage 1 works in terms of performance was measured on `LexicalStore` on main. This is not a measurement on PostgreSQL. A wrapper that counts storage operations was inserted, and a storage that does not return `as_slice` was used (the same condition as a table-backed storage).

**Conditions**: 200k documents, 60 tokens each, `body` index-only (`stored = false`), `ctid` as a stored integer. File storage (buffered file I/O, no mmap). **One thread** (the extension runs without threads, so rayon was limited to one thread). Two cases: a large vocabulary (200k terms) and a small one (20k terms). 20 to 24 segments. The query is an OR of three terms, top 10.

| Item | Vocabulary 200k | Vocabulary 20k |
| :--- | :--- | :--- |
| Index size | 239 MB | 98 MB |
| One search (warm, no `as_slice`, 1 thread) | 1.9 ms | 3.5 ms |
| Same, 2 threads (for reference) | 1.3 ms | 2.8 ms |
| Same, 2 threads, mmap with `as_slice` (for reference) | 0.49 ms | — |
| Files opened / bytes read per search | 60 opens / about 12 KB (about 66 chunks of 2 KB) | 54 opens / about 38 KB (about 73 chunks) |
| First search right after opening (1 thread) | 2.28 s, reads 90 MB | 0.19 s, reads 13 MB |
| First search after rebuilding the store (1 thread) | 1.97 s | 0.16 s |
| Growth in process memory | about 790 MB | about 140 MB |

**A commit of a single document** (add one document to an index with 20 segments and commit):

- Two files are written (the segment `.cfs`, about 4 KB, and `segments.json`, about 3 KB). Three creates and two renames.
- 0.6 to 0.9 ms on memory storage.
- `segments.json` grows in proportion to the number of segments. Averaged over 100 commits growing to 120 segments, about 15 KB was written per commit.

**What this shows**:

- **Search speed is within a practical range.** 2 to 4 ms on one thread without `as_slice`. At the same number of threads, the path without `as_slice` is about 2.7 times slower than mmap (1.3 ms versus 0.49 ms on two threads). A table-backed storage adds chunk fetches on top (60 to 70 rows per search).
- **The cost right after opening is decode, not I/O.** The amount read is 13 to 37% of the index, but most of the time is spent building term dictionaries. It is proportional to vocabulary size × number of segments.
- **Rebuilding the store costs as much as opening it.** This is the basis for S5 being required.
- **Memory is loaded per backend.** With a large vocabulary it exceeds the index size (about 790 MB for a 239 MB index). It grows in proportion to the number of connections. This does not depend on the storage approach; it comes from readers holding decoded structures on the heap.
- **A single-document commit is about 1 ms plus a few row INSERTs.** Even serialized, several hundred documents per second should go in. But every commit adds a segment, and the number of files a search opens (terms × segments) grows too.

**Measurement note**: `MemoryStorage::open_input` copies the whole file on every open (`storage/memory.rs`). The path without `as_slice` opens files on every search, so on memory storage a search appears to take 30 ms. The table above uses file storage.

### 4.2 Limits

| Limit | Content |
| :--- | :--- |
| Serialized writes | While a transaction writing to the index is open, other writes to that index wait. They also wait during merges |
| Memory per backend | Reader structures are loaded per connection. Lexical is proportional to vocabulary × number of segments (4.1). Vector cannot return `as_slice`, so it becomes `Eager` (copied to the heap): about 1.5 GB with SQ8 and about 3 GB with Float16 for 1M vectors of 1536 dimensions |
| Delay right after connecting | The first search builds term dictionaries and other structures (0.2 to 2.3 s per 200k documents in 4.1) |
| I/O path | Every range read looks up the primary key of an auxiliary table |
| Small writes | Every statement creates one segment. Vector builds an HNSW graph per segment |
| Serial computation | The build has no threads, so both search and index construction run on one core |
| doc_id limit | Posting encoding reduces the absolute doc_id to u32 (`inverted/core/posting.rs`). doc_ids are not reused, so an index can issue about 4.3 billion over its lifetime. Each UPDATE consumes one. `REINDEX` resets it |

The limits in 2.11 apply as well (false positives under SERIALIZABLE, no benefit from `CREATE INDEX CONCURRENTLY`, unlogged tables not supported, interaction with `FOR ALL TABLES` publications).

**Uses that work and uses that do not**:

- **Work**: Read-mostly indexes where writes are batched or sparse. Deployments where a pooler keeps the number of connections small. Lexical indexes of a few hundred thousand to a few million documents.
- **Hard**: Indexes with many connections and a large vocabulary (memory multiplies by the number of connections). Tables with many concurrent writers. Vector indexes of several million or more (the per-backend copy and serial HNSW construction).

### 4.3 What to measure on the prototype

| Measurement | Which later item it informs |
| :--- | :--- |
| Wait time under concurrent writes | Catalog (chapter 5) |
| Delay of the first search after someone else's commit (with S5 in place) | Following by version in the Catalog (C2) |
| Time of one search on the auxiliary tables (difference from the file storage values in 4.1) | Range reads, direct placement in pages (chapter 6) |
| Total memory as the number of connections grows | Sharing reader memory (chapter 6) |
| Number and time of auxiliary table reads per search | Range reads, lazy posting decode, direct placement in pages (chapter 6) |
| Memory per backend (against the number of segments and documents) | Reader memory (chapter 6) |
| Growth in the number of segments and merge time | Absorbing small writes (chapter 6) |
| VACUUM duration and lock hold time | Splitting the application of deletions (2.5) |

### 4.4 PostgreSQL-side points to verify during stage 1

- Writing to auxiliary tables from inside `aminsert` and `aminsertcleanup`. Citus columnar does this from a table AM (10.2). No precedent in an index AM has been confirmed.
- The switch of storage id and the deletion of old rows under `REINDEX`, `TRUNCATE`, `DROP INDEX` and `CREATE INDEX CONCURRENTLY`.
- Handling of indexes on unlogged / temporary tables (the auxiliary tables are ordinary tables, so the persistence does not match).
- VACUUM of the auxiliary tables themselves (file deletions produce dead tuples) and the degree of bloat.
- Functions that report index size, such as `pg_relation_size` (most of the data is in the auxiliary tables).
- Parallel scan workers each opening a store (memory multiplies by the number of workers). Parallel scan is not supported at first.

---

## 5. Stage 2: concurrent writes through a Catalog

Remove write serialization and keep reader caches across commits. The core changes are concentrated here.

### 5.1 A table-backed Catalog

- The segment list is kept in a table separate from the auxiliary tables, with one row per segment.
- Publishing is a row INSERT, a merge is a DELETE of the sources plus an INSERT of the merged segment, and an update of deletion state is a row UPDATE.
- Readers see the list through a snapshot, so they get a consistent set without comparing versions. Reader caches are reused per segment.
- When two writers publish different segments, the rows differ, so they do not conflict. When two update the deletion state of the same segment, they are serialized by a row lock, and the later one reads the latest state and takes the union.

### 5.2 Change items

| # | Content | Current code (confirmed) |
| :--- | :--- | :--- |
| **C1** Catalog trait | Put the lexical `segments.json`, the per-field vector `segments.json`, numbering, the checkpoint and the deletion state version behind one trait. The default (local) implementation reproduces today's behavior | `inverted/segment_manifest.rs`, `vector/index/segment/manager.rs` |
| **C2** Following by version | Make `InvertedIndex::refresh` and `VectorIndex::refresh` re-read when the Catalog has advanced. One search uses the set it obtained to the end | `InvertedIndex::refresh` does not re-read (#1023). `VectorIndex::refresh` is a no-op by default |
| **C3** Moving numbering | The Catalog issues segment name numbers | Lexical uses `reserve_generation` and the writer's `current_segment` (in memory). Vector uses `generate_segment_id` |
| **C4** Explicit sweep | Turn off the orphan sweep at open with an option; only the maintenance owner calls it | `sweep_orphans` in lexical, `cleanup_orphans` in vector |
| **C5** Split commit | Make flush, persisting deletions, publish, checkpoint, sync and merge callable one by one | The lexical writer runs flush → `flush_deletions` → `publish_pending_segments` → `write_metadata_json`, and the order matters. In vector, seal writes the file and registers it in one step |
| **C6** Single merge execution | A merge runs only after taking the right to run. List the places that delete files | Source files are deleted right after the merge |
| **C9** Merge and deletion consistency | Carry deletions that entered the sources during a merge over to the merged segment. With the stage 1 structure (deletion marks in index pages, folded into Laurus), a failed carry-over still leaves search results correct; documents that should disappear merely stay in the segment. It becomes a matter of space and statistics, not correctness | Lexical rebuilds from the `.delmap` as of the start. Exclusion is an in-process lock. In vector the bitmap is keyed by doc_id and there is one for the whole index, so deletions during a merge remain (#1518) |
| **L1** Manifest and metadata | Move `last_wal_seq` in `metadata.json` to the Catalog. `doc_count` and others are already derived from the manifest (#1212) | `inverted.rs` |
| **L2** Multiple writers | Hand out independent writers. Make the writer's segment cache follow the version. Add an append-only entry point for new doc_ids that skips the lookup of old versions | There is one `writer_cache`. `segment_ranges` and `DeletionManager` stay as built. The `max_committed_doc_id` comparison does not work when doc_ids arrive out of order |
| **L3** `.delmap` races | Check the version; if it has advanced, take the union with the latest and rewrite | The per-segment `.delmap` is replaced by temp + rename. Lexical deletions only grow within a segment |
| **V1** Save-then-swap | Align to "build a candidate → save → apply to memory on success" before moving to the Catalog | Mutate-then-save |
| **V2** Publishing across fields | Combine all fields and lexical into one publish | Each field has its own manifest (`multi_field.rs`) |
| **V3** Deletion bitmap | Check the deletion version. `forget_deletions` happens together with the merge publish | The safety condition of `forget_deletions` is the in-process writer lock. Only multivector calls `unmark_deleted` |
| **V4** Sync entry point and multiple writers | Hand out sync writers for precomputed vectors. Add an append-only entry point | `write_prepared` calls `delete_document` before adding |

### 5.3 Catalog trait draft (proposal)

```rust
pub trait Catalog: Send + Sync + std::fmt::Debug {
    /// The current version. Readers call it on every search, so it must be cheap.
    fn version(&self) -> Result<u64>;

    /// The current state and its version. One read returns a consistent set.
    fn load(&self) -> Result<(u64, CatalogState)>;

    /// Apply a list of operations to the latest state and publish.
    /// Either the whole list takes effect at once, or nothing changes.
    fn publish(&self, ops: &[CatalogOp]) -> Result<u64>;

    /// Reserve n segment numbers that do not collide.
    fn allocate_segment_ordinals(&self, n: u32) -> Result<std::ops::Range<u64>>;

    /// The right to run maintenance (merge, applying deletions). None if not available.
    fn try_claim(&self, scope: MaintenanceScope) -> Result<Option<MaintenanceClaim>>;
}

pub enum CatalogOp {
    AddLexicalSegment(SegmentInfo),
    AddVectorSegment { field: String, info: ManagedSegmentInfo },
    /// Result of a merge. Fails if any source is missing from the current state.
    ReplaceLexicalSegments { sources: Vec<String>, merged: Vec<SegmentInfo> },
    ReplaceVectorSegments { field: String, sources: Vec<String>, merged: ManagedSegmentInfo },
    /// Update of deletion state. Fails if expected differs from the current version.
    SetLexicalDeletion { segment_id: String, expected: u64, deleted_count: u64 },
    SetVectorDeletion { field: String, expected: u64 },
    SetCheckpoint(u64), // used only by Engine
}
```

- A list of operations, not a closure over the whole state. In the table implementation each operation maps to a row INSERT / DELETE / UPDATE, and a retry just applies the same list again.
- `publish_with` already takes "a delta to the list", and there is a test stating that a merge delta must be applied to the live list (`publish_with_applies_deltas_to_the_live_list`). A list of operations extends that.
- Three implementations are expected: local, reproducing today's behavior (the default of `Engine`); table (the PostgreSQL extension); and object storage (the same thing as multi-writer coordination for Bundle).
- The local implementation reads and writes the existing `segments.json` (lexical v2, vector v1) as is and does not change the format.

### 5.4 How to do C9

- **Exclusion**: Never run the application of deletions and a merge on the same segment at the same time. The pg_textsearch way (wait until whoever entered first finishes).
- **Carry-over**: At merge publish, re-read the sources' `.delmap` and copy the marks added since the start to the merged segment. The same as a merge commit in Lucene. doc_ids do not change in a merge, so a union of bitmaps is enough.
- **Recommendation**: Carry-over. Exclusion makes VACUUM and merges wait for each other for a long time.

### 5.5 Order

1. V1 (save-then-swap; can go in without the Catalog)
2. C1 + L1 (the Catalog trait and the local implementation; behavior unchanged)
3. C2 + C3 + C4 (lexical)
4. C5 + L2 + L3 + C9 (lexical)
5. Apply to vector (C2 / C3 / C4, V3, V4)
6. C6 + V2

The condition at the end of every step is that all existing tests pass.

---

## 6. Stage 3: performance

Start with what the stage 1 measurements (4.3) show to be needed.

| # | Content | Current state (confirmed) | Direction of change |
| :--- | :--- | :--- | :--- |
| **C10** Absorbing small writes | A layer that takes per-statement segments cheaply | The lexical merge rebuilds by replaying postings. Vector builds an HNSW graph per segment and a merge rebuilds from all vectors | Vector: segments under a threshold get no graph and are searched by brute force, then turned into HNSW by a merge. `SegmentedIndex<B>` fixes the backend as a type parameter, so record the segment kind and allow a mix. Lexical: handle with merge policy first |
| **C11** Positioned range reads | Reads that do not move a cursor | `StorageInput` is `Read + Seek` with `clone_input` and an optional `as_slice` (whole file) | Add `read_at(offset, buf)`. The default implementation is `clone_input` + seek + read |
| **V6** Reader memory | Do not load the graph and rows onto the heap | The graph is `Vec<Vec<Vec<u32>>>`. Rows are `Eager` (copied) or `Lazy` (needs a slice of the whole file). PQ codes are on the heap in both modes | Put rows on top of range reads (the position arithmetic of `MappedRows` can be reused). Read adjacency lists from the file and keep only the node → offset table in memory |
| **F1** Lazy posting decode | Read only the blocks that are needed | `SegmentReader::postings` decodes the whole list per term. Block-Max WAND skips over the decoded array, so I/O does not decrease | The skip table and block max are already in the file. The change is mainly in the reader |
| **F2** Resident dictionary and others | Read in place with range reads | Opening a segment loads the FST and block section in full. `.docs` is also loaded in full on first reference | Move the format towards in-place reading |
| **F3** Segment-relative doc_id | Make the u32 limit per segment | The absolute doc_id is reduced to u32 | Store values relative to the segment. The carry-over of C9 needs a mapping. The stage 1 doc_id → ctid array assumes a doc_id unique across the index, so a logical id must be kept separately (1.4) |
| **F4** HNSW graph block | Keep an offset table in the file | The v2 block is variable-length per node and has no offset table | Done when V6 does not choose to build the table at load time |
| **P1** Direct placement in pages | Do not go through auxiliary tables | — | Only when measurements show I/O to be a problem. The hard parts on the extension side (chapter 8) have to be written by hand |

F1 to F4 are breaking changes to the on-disk format. They are designed together with the Bundle format.

**Precedents in reference implementations**:

- Small writes: pg_search has mutable segments (segments that only append a list of ctids; frozen at a threshold or by VACUUM), pg_textsearch has a memtable chain on pages, and in pgvector the graph itself is mutable.
- Two tiers for vector: the Tantivy fork in pg_search has a `.vec` per segment and builds `.centroids` only for IVF segments.
- Range reads: the pg_search reader does not load whole files; it reads pages with `FileHandle::read_bytes(range)`.
- Segment-relative doc_id: pg_textsearch uses numbers local to the segment as doc_ids.

---

## 7. Verification

### 7.1 Stage 1

| Property to check | Procedure |
| :--- | :--- |
| Builds without threads | Add a native build and tests with `--no-default-features` to CI |
| Works without `_id` | Add, search, delete and merge on an index without an `_id` field |
| Works without stored fields | A document with no stored fields shows up in search after a merge |
| Merge can be stopped | After a commit with the switch off, the number of segments does not change. Calling merge separately reduces it |
| Works without `as_slice` | All lexical and vector tests pass on a storage whose `as_slice` always returns `None` |
| Manifest reload | After another store commits, a reload makes the new segment searchable. Readers of unchanged segments are not rebuilt (checked by counting storage reads). After a merge, a reload does not reference removed segments |
| doc_id counter | No past doc_id is issued to a new document after deleting the last documents and merging, or after an abort |
| Nothing remains after abort | Abort the transaction after writing. Nothing remains in the auxiliary tables or in search results |
| Savepoint | Write inside a savepoint and roll back. Later searches and writes are correct |
| Others' commits are visible | Backend A commits. They show up in backend B's next statement |
| Old snapshots can read | While backend A is searching, B commits a merge. A's search completes |
| VACUUM | VACUUM after DELETE. The row does not show up in search, and a reused ctid does not return another row. No write to the auxiliary tables happens during VACUUM |
| Array after abort | Abort a write. The array entry remains but does not show up in search, and a later VACUUM marks it |
| Delete folding | After marked documents are passed to Laurus deletion, a merge removes the data and search results do not change |
| Read-back within a statement | Insert several rows in one statement; reads and writes are correct after a flush in the middle |
| Writer snapshot | A's write waits for B's lock. B commits. A then commits. B's documents show up in search. Checked under both READ COMMITTED and REPEATABLE READ |
| Recovery after errors | Cause a lock timeout or cancel in the middle of a write. The next statement works correctly |
| DDL | After `REINDEX`, `TRUNCATE` and `DROP INDEX`, no rows of the old storage id remain. After rolling back the DDL, the original index is usable |
| `CREATE INDEX CONCURRENTLY` | INSERTs during the build are in the index after the build |
| Privileges | Ordinary users cannot read the auxiliary tables |
| Standby | Search works on a standby and reflects updates on the primary |

### 7.2 Stage 2

Create two stores (A, B) that point at the same storage and check with a Catalog that imitates sharing between processes.

| Property to check | Procedure |
| :--- | :--- |
| Publishes are not lost | A and B alternate flush + publish. All segments remain |
| Publishes are visible | A publishes. B sees it in a search after refresh |
| The set is fixed during a search | A starts a search. B publishes a merge. A completes with the set as of the start |
| Names do not collide | A and B flush at the same time. Segment names do not overlap |
| Unpublished files are not deleted | A flushes (unpublished). B opens. A's files remain |
| Deletions are not lost (lexical) | A and B delete different docs of the same segment. Both marks remain |
| Another writer's segment can be deleted | A doc in a segment published by A can be deleted by a writer of B that has been alive since before |
| Deletions during a merge are not lost | A starts a merge. B deletes a doc in a source and publishes. A publishes the merge. The doc does not show up in search |
| Deletions do not come back (vector) | A merges all segments + forget. At the same time B deletes another doc. The reclaimed marks do not come back and B's mark remains |
| Merges do not run twice | A and B try to merge at the same time. Only one runs |
| Existing behavior | All existing tests pass with the local Catalog. In particular the regression tests for #864, #875, #1021, #1023, #1207, #1499, #1518 and #1527 |

---

## 8. Hard parts of placing data directly in pages (for reference)

Not handled in stages 1 and 2. Extracted from the reference implementations for when P1 is considered.

| Item | Content | Reference |
| :--- | :--- | :--- |
| Page reclamation | No immediate reuse. Decide by the xid horizon or by whether a pin exists | pg_search (garbage list, pin check, free space map with xids), pg_textsearch (`GetOldestNonRemovableTransactionId`) |
| Standby | Emit a conflict record before reuse, or require `hot_standby_feedback`. If not supported, refuse reads on a standby | pg_textsearch, pg_search Community edition |
| WAL order | Emit the record of a new page before the update of the pointer to it | pg_search (`linked_items.rs`) |
| `GenericXLog` limit | 4 pages per record | PostgreSQL (`XLR_NORMAL_MAX_BLOCK_ID`) |
| Merge and VACUUM | Prevent deleted entries from coming back by ordering or by an exclusion list | pg_textsearch, pg_search |

---

## 9. Current state (appendix)

Places that assume ownership by a single process. They are the basis for the stage 2 changes. All confirmed on main.

### 9.1 Lexical

| Place | Current state |
| :--- | :--- |
| `inverted/segment_manifest.rs` `publish_with` | Clone the in-memory segment list → apply the delta → save → swap into memory |
| `inverted.rs` `InvertedIndex::refresh` | Does not re-read storage (#1023) |
| `segment_manifest.rs` `next_generation`, `writer.rs` `current_segment` | Segment name numbers come from in-memory counters |
| `inverted.rs` `sweep_orphans` | At open, deletes segment files that are not in the manifest |
| `lexical/store.rs` `writer_cache` | One writer |
| `lexical/store.rs` `commit_with_checkpoint` → `finish_commit` | Runs writer commit, sync, auto-merge and refresh in sequence |
| `writer.rs` `flush_deletions`, `maintenance/deletion.rs` | Replaces the per-segment `.delmap` by temp + rename |
| `writer.rs` `segment_ranges`, `deletion_manager` | A copy of the committed segments as of construction. Updated only by `invalidate_segment_cache` |
| `writer.rs` `rollback` | Discards unpublished segments. Does not undo deletions on committed segments |
| Merge and deletion | `commit` drops the writer before merging. `optimize` holds the writer cache lock |

### 9.2 Vector

| Place | Current state |
| :--- | :--- |
| `segment/manager.rs` `add_segment` and others | Mutate-then-save |
| `manager.rs` `generate_segment_id` | In-memory counter (the value is saved in the manifest) |
| `manager.rs` `cleanup_orphans` | Inside `SegmentManager::new`, deletes segment files that are not in the manifest |
| `vector/index.rs` `refresh`, `segmented_index.rs` `searcher` | Searchers are built from the in-memory segment list and bitmap |
| `multi_field.rs` | Each field has its own manifest and `.delmap` |
| `segmented_core.rs` | One deletion bitmap per field, keyed by doc_id. Shrinks with `forget_deletions`. `write_delmap` rewrites the whole bitmap every time |
| `segmented_index.rs` | One commit = one segment. The backend is fixed as a type parameter |
| `hnsw/segment/merge_engine.rs` | Reads all vectors and rebuilds with a new writer. The writer has a path that adds to an existing graph, but merge does not use it |
| `vector/store.rs` | Writes are async. `write_prepared` is `delete_document` → add |
| `format.rs` | Rows in `.hnsw` are SQ8 / PQ / PQ FastScan / Float16. Raw f32 is in the rerank sidecar (`.hnsw.f32`) |

---

## 10. Key points of the reference implementations (appendix)

### 10.1 Full-text and vector index AMs

| Topic | pg_search | pg_textsearch | pgvector |
| :--- | :--- | :--- | :--- |
| Where data lives | Index pages (linked lists of blocks) | Index pages (memtable chain and segment chain) | Index pages (element tuples and neighbor tuples) |
| When writes are published | End of statement (`aminsertcleanup`) | At `aminsert` | At `aminsert` |
| Segment visibility | Not tied to transactions. The heap decides | Same | No segments |
| Small writes | Mutable segments | Memtable chain (the shared memory cache is derived) | The graph is mutable |
| Merge and VACUUM | Merge lock, vacuum list, cancel requests | Wait until whoever entered first finishes | Repair the graph, then delete |
| Page reclamation | Pin check + free space map with xids | xid horizon + conflict record | Detect reuse with the tuple `version` |
| Standby | The Community edition refuses reads | Requires `hot_standby_feedback` | Supported |
| Reader | Range reads (`FileHandle::read_bytes`) | Pages are the source of truth | Walks pages directly |
| Threads | Serial writer | — | Parallel build uses shared memory + relative pointers |

All three keep their data in the index's own pages. None uses auxiliary tables.

### 10.2 A precedent for auxiliary tables: Citus columnar

Citus columnar is a table AM, not an index AM, but it is a precedent for reading and writing other tables from inside access method callbacks (`columnar_metadata.c`, `columnar_tableam.c`).

| Topic | How Citus columnar does it | How stage 1 reflects it |
| :--- | :--- | :--- |
| How data is split | Bulk data (stripes) in the relation's own pages. Metadata on which stripes exist in table rows | Laurus files in tables; what VACUUM writes in index pages (2.1) |
| How tables are held | Shared tables owned by the extension (`columnar.stripe`, `columnar.chunk_group`, `columnar.chunk`), looked up by storage id | The same (2.2) |
| Writing rows | Executor helper functions (`ExecSimpleRelationInsert`, `simple_heap_delete`). No SQL | The same (2.3) |
| Reading back own writes | `CommandCounterIncrement()` after writing. `GetTransactionSnapshot()` for reads | The same (2.3) |
| Cleanup on deletion | `DeleteMetadataRows` deletes the rows of a storage id | The same (2.2) |
| Lazy VACUUM | Only truncates free space at the end. Does not update the metadata tables (comment: there are no updates or deletes, so truncation is enough) | VACUUM does not write to the auxiliary tables (2.8) |

No precedent was found for writing auxiliary tables from lazy VACUUM. PostgreSQL itself states that this must not be done (2.8).

### 10.3 Extensions that abstract storage

Extensions built as "a search implementation placed on PostgreSQL through a storage trait", and the unit of their abstraction.

| Extension | Unit of abstraction | Content |
| :--- | :--- | :--- |
| pg_search | **File** (Tantivy's `Directory` trait) | Tantivy abstracts storage behind `Directory`. pg_search implements it on PostgreSQL blocks (`MVCCDirectory`, and `BuildDirectory` for builds). This is the same layer of abstraction as Laurus's `Storage` trait and the closest structure |
| VectorChord | **Page** (traits such as `Page`, `RelationRead`, `RelationWrite`) | The search algorithms (`crates/vchordrq`, `crates/vchordg`) are written against page and guard traits. Besides the PostgreSQL implementation (`PostgresRelation`) there are implementations with a cache for builds, with buffering, and with hooks for maintenance |
| Lantern | **Node retrieval** (usearch callbacks) | Writes an index built by usearch into pages and, at search time, gives usearch a callback that retrieves nodes from pages |
| pgvectorscale | Node layout (`Storage` trait) | Switches between a layout without quantization (plain) and SBQ. The backing store is always PostgreSQL pages; it does not abstract where data lives |
| pgvecto.rs | None | The index is its own mmap files. A background worker owns them and backends query it over IPC. The README says WAL support for indexes is work in progress and recommends moving to the successor VectorChord (on pages) |
| PGroonga | None | Uses Groonga's own files. WAL and crash handling are its own mechanisms (`pgroonga-wal-resource-manager` and others). Only the file layout was looked at; details were not read |
| pgvector, pg_textsearch | None | Operate on pages directly |

**What this shows**:

- There are two kinds of abstraction: by file (pg_search) and by page (VectorChord). Laurus's `Storage` is by file, so direct placement in pages (P1 in chapter 6) would follow the pg_search kind.
- In every extension the final home of the data is the index's own pages. No index AM keeps it in tables. Stage 1 uses tables as the step before that.
- The structure with own files plus a background worker (pgvecto.rs) was rebuilt on pages by the same developers. Not being covered by WAL and replication was its weakness.
- Outside PostgreSQL, keeping a full-text index in the host database's tables is an established design. SQLite FTS5 stores segment pages as rows of a shadow table (`%_data`, keyed by a rowid computed from the segment id and page number) with a per-segment term index in `%_idx`. MySQL InnoDB full-text search keeps the inverted index in six auxiliary index tables and batches small insertions in an in-memory cache before flushing to them. Both were checked in their documentation and source comments, not by reading the implementation. The difference from stage 1 is granularity: FTS5 and InnoDB designed their formats for tables and can read one page or one word per row, while stage 1 cuts a file-oriented format into chunks. Lucene also had a `Directory` implementation on RDBMS tables; that is from memory and was not checked.

---

## 11. Overall order

### 11.1 Developing the extension (stage 1)

At the end of each milestone, the corresponding tests (7.1) pass.

| Milestone | Content | What to verify |
| :--- | :--- | :--- |
| **M0** Core groundwork | S1 (thread feature gate), S2 (public APIs), S4 (merge switch), S5 (manifest reload). S1 can ship on its own | Existing tests pass. A native build without threads passes |
| **M1** Skeleton | The pgrx crate, auxiliary tables, metapage, TableStorage reads and writes | That index AM callbacks can write auxiliary tables (the largest uncertainty, with no confirmed precedent). That Laurus tests pass without `as_slice` |
| **M2** Build and search | `ambuild`, `amgettuple`, the array, a minimal operator | Search results match Laurus on its own. Time of one search on the auxiliary tables |
| **M3** Writes and transactions | `aminsert`, `aminsertcleanup`, the write lock, the snapshot rules, abort, savepoint | Every path in 2.6. The writer snapshot test (READ COMMITTED and REPEATABLE READ) |
| **M4** VACUUM and worker | `ambulkdelete`, delete folding, merge | A reused ctid does not return a wrong row. No write to the auxiliary tables during VACUUM |
| **M5** DDL and operations | `REINDEX`, `TRUNCATE`, `DROP INDEX`, privileges, recovery from errors, standby | No old rows remain. Search works on a standby |
| **M6** Measurement | The items in 4.3 and the PostgreSQL-side checks in 4.4 | Material for deciding whether to go to stage 2 or stage 3 |
| **M7** Vector | Put `VectorStore` on the same structure (2.10) | Memory per backend, time of serial HNSW construction |

If M1 shows that callbacks cannot write to auxiliary tables, this approach is dropped and placing data in the index's own pages (P1 in chapter 6, chapter 8) is considered.

**Where the crate lives**: `laurus-postgresql` is added to the `mosuka/laurus` repository. It is excluded from the default workspace build (it needs a PostgreSQL toolchain), has its own version separate from `[workspace.package]`, and is built and tested in a separate CI job.

### 11.2 After that (including the other initiatives; reasons in 1.4)

| Step | Content | Where |
| :--- | :--- | :--- |
| 1 | **Format design**: decide Bundle, the column store, the format changes (F1 to F4), how doc_ids are held, and the representation of pushdown in one pass | 1.4, chapter 6 |
| 2 | **Implement object storage and the column store**. The Catalog trait is built here | 5.3 |
| 3 | **SQL support** | 1.4 |
| 4 | **Stage 2 and later of the PostgreSQL extension**. Chosen by the results of M6 | Chapters 5 and 6 |

How to choose in step 4:

- Waits under concurrent writes, or waits during merges, are the problem → stage 2 (chapter 5).
- I/O, memory per backend, or the number of segments is the problem → the matching items of stage 3 (chapter 6). From the results in 4.1, memory (V6 and F2) is expected to matter first.

---

## 12. Unverified items

### Related to stage 1

The milestone in which each is verified (11.1) is in parentheses.

1. Writing to auxiliary tables from inside `aminsert` / `aminsertcleanup` of an index AM. A table AM (Citus columnar) is a precedent, but none has been confirmed for an index AM. (M1)
2. Whether S5 (manifest reload) is as small a change as expected. The relevant code was read but not implemented. (M0)
3. Read speed through the auxiliary tables. 4.1 is on file storage; nothing was measured on tables. (M2)
4. Whether Laurus has an entry point that returns all matching doc_ids, for `amgetbitmap`. (M2)
5. A placement that shares analyzer dictionaries between processes (mmap). (M2)
6. Safe recovery through pgrx unwinding when a PostgreSQL error occurs in the middle of a Laurus call (2.11). (M3)
7. The share of the top k dropped by deletion marks and heap visibility, and how often a re-search happens. (M4)
8. The switch of storage id and the deletion of old rows under `REINDEX`, `TRUNCATE`, `DROP INDEX` and `CREATE INDEX CONCURRENTLY`. How to catch `DROP INDEX` (object access hook or event trigger) and whether anything is missed. (M5)
9. The frequency of serialization failures under SERIALIZABLE (2.11). Inferred from the rules, not verified by running. (M5)
10. That a `FOR ALL TABLES` publication includes the auxiliary tables (2.11). Based on reading the deciding function in PostgreSQL, not verified by running. (M5)
11. Vocabulary size and memory with real documents. The measurements in 4.1 use synthetic data (vocabulary skew produced by a simple formula). (M6)
12. The size of the doc_id → ctid array. Entries are added on every update and it does not shrink until `REINDEX`. (M6)
13. WAL volume (2.11). (M6)
14. The cost of the deletion log (which uses `create_output_append`) when implemented by adding rows on every append. It can be turned off by a setting (`enable_deletion_log`). (M6)
15. HNSW construction time in a build without threads. (M7)
16. The meaning of `LexicalSearchResults.total_hits`. In a measurement, a query matching 3000 documents across 3 segments returned 15 with `limit(5)` and 3000 with `limit(100000)`. Under top-k collection it looks like "the sum of the numbers collected per segment". Whether this is by design or a defect has not been checked. It cannot be used when a match count is needed. (M2)

### Related to stage 2 and later

17. What happens to loading the rerank sidecar (`.hnsw.f32`) on a storage that cannot return `as_slice`.
18. Whether the incremental path of the HNSW writer is good enough to be the base of a merge. A comment says that adding to a small base needs a warm-up.
19. How IVF segments fit into the two-tier structure of C10.
20. The amount of change needed to put a multivector index on the Catalog (it has its own implementation that uses `SegmentedCore` and `SegmentManager` directly).
21. Memory used by the lexical merge replay on large segments (that it is split by `max_merge_buffer_memory` was confirmed).
22. The implementation of standby support in pg_search (Enterprise edition; not in the public code).

---

## 13. Revision history

| Version | Content |
| :--- | :--- |
| 1 | A list of changes that let `LexicalStore` / `VectorStore` be used from multiple processes (the Catalog trait and others) |
| 2 | Verified against the code on main and reference implementations (pg_search, pg_textsearch, pgvector, PostgreSQL itself). Added the race between merge and deletion (C9) and others, and corrected errors |
| 3 | Changed the plan to stages. Start with a Storage that keeps files in PostgreSQL tables |
| 3, revision 1 | Measured stage 1 performance (4.1). Found that reopening the store does not work and added the manifest reload (S5) |
| 3, revision 2 | Reviewed against similar extensions (Citus columnar, VectorChord and others) and PostgreSQL itself. Found that writing auxiliary tables from VACUUM was wrong, and moved the doc_id → ctid mapping and the deletion marks to the index's own pages. The change that kept the ctid in a Laurus column (S3) became unnecessary |
| 3, revision 3 | Checked for omissions. Fixed the snapshot a writer uses, rejection of unlogged tables, the role of delete folding, storage cleanup, privileges on the auxiliary tables, and the kind of lock (2.11) |
| 4 | Merged the above into the body. Added the order relative to the other initiatives (columnar doc store, object storage, SQL) (1.4, chapter 11) |
| Merged | Merged the extension development plan into this document. Added the scope (1.5), data structures (2.12), the callback table (2.13), the background worker (2.14), DDL (2.15), the SQL surface (2.16), and the milestones M0 to M7 (11.1) |
| Merged, final review | Re-measured on one thread (4.1; earlier values were on two threads). Added the read-only rule for reader stores (2.3), when `version` advances (2.12), the format version and memory management (2.11), and the ordering operator (2.16) |
| English edition | Translated to English and added to the repository. Crate name set to `laurus-postgresql`. Added the read batching and chunk cache rules (2.3), where the crate lives (11.1), and the comparison with SQLite FTS5 and MySQL InnoDB full-text search (10.3) |
