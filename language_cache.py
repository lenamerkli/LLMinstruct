import hashlib
import json
import sqlite3
import typing as t


class LanguageCache:
    def __init__(self, db_path: str = 'language_cache.db'):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        self._conn.execute('CREATE TABLE IF NOT EXISTS cache (hash TEXT PRIMARY KEY,languages TEXT)')
        self._conn.commit()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> t.Optional[t.List[str]]:
        hash_value = self._hash_text(text)
        cursor = self._conn.execute(
            'SELECT languages FROM cache WHERE hash = ?',
            (hash_value,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row['languages'])

    def set(self, text: str, languages: t.List[str]) -> None:
        hash_value = self._hash_text(text)
        self._conn.execute(
            'INSERT OR REPLACE INTO cache (hash, languages) VALUES (?, ?)',
            (hash_value, json.dumps(languages))
        )
        self._conn.commit()

    def get_batch(self, texts: t.List[str]) -> t.List[t.Optional[t.List[str]]]:
        hashes = [self._hash_text(text) for text in texts]
        placeholders = ','.join('?' * len(hashes))
        cursor = self._conn.execute(
            f'SELECT hash, languages FROM cache WHERE hash IN ({placeholders})',
            hashes
        )
        cache_map = {row['hash']: json.loads(row['languages']) for row in cursor.fetchall()}
        return [cache_map.get(h) for h in hashes]

    def set_batch(self, texts: t.List[str], languages_list: t.List[t.List[str]]) -> None:
        data = [
            (self._hash_text(text), json.dumps(languages))
            for text, languages in zip(texts, languages_list)
        ]
        self._conn.executemany(
            'INSERT OR REPLACE INTO cache (hash, languages) VALUES (?, ?)',
            data
        )
        self._conn.commit()

    def detect_with_cache(
        self,
        detector,
        texts: t.List[str]
    ) -> t.List[t.List[str]]:
        # Get cached results
        cached_results = self.get_batch(texts)

        # Find uncached texts
        uncached_indices = [i for i, result in enumerate(cached_results) if result is None]
        
        if uncached_indices:
            # Run detection only on uncached texts
            uncached_texts = [texts[i] for i in uncached_indices]
            detection_results = detector.detect_multiple_languages_in_parallel_of(uncached_texts)

            # Process and cache new results
            new_languages_list = []
            for results in detection_results:
                languages = list(set(result.language.iso_code_639_1.name for result in results))
                new_languages_list.append(languages)

            # Store new results in cache
            self.set_batch(uncached_texts, new_languages_list)

            # Merge cached and new results
            for idx, languages in zip(uncached_indices, new_languages_list):
                cached_results[idx] = languages

        return cached_results

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> 'LanguageCache':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()