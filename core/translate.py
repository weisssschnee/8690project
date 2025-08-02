import streamlit as st
import functools
import logging
import json
from pathlib import Path
import os
import time
from typing import Optional, Dict, List  # Ensure all needed types are imported

# Try importing google_trans_new
try:
    import google_trans_new

    GOOGLETRANS_AVAILABLE = True
    logger_init_trans = logging.getLogger(__name__)  # Use specific logger name
    logger_init_trans.info("Successfully imported google_trans_new.")
except ImportError:
    google_trans_new = None  # Define as None if import fails
    GOOGLETRANS_AVAILABLE = False
    logging.warning("google_trans_new library not found. Automatic translation feature will be disabled.")

# Configure logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Uncomment for more verbose logs
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class AutoTranslator:
    """
    Hybrid translator class: Combines local pre-defined translations,
    auto-translation cache, and online translation.
    """

    def __init__(self, locales_dir="locales", auto_cache_dir="locales/auto_cache", default_lang="zh-cn",
                 default_mode="local_then_auto"):
        """
        Initializes the translator.
        Args:
            locales_dir (str/Path): Directory for local translation files relative to project root.
            auto_cache_dir (str/Path): Directory for auto-translation cache relative to project root.
            default_lang (str): Default language code ('zh-cn', 'en').
            default_mode (str): Default translation mode ('local_only', 'local_then_auto').
        """
        logger.info(f"Initializing Hybrid Translator...")

        # --- Robust Path Resolution ---
        try:
            # Assume translate.py is in project_root/core/translate.py
            # Thus, project_root is parent of parent of this file
            self.project_root = Path(__file__).parent.parent.resolve()  # Get absolute path of project root
            self.locales_path = (self.project_root / locales_dir).resolve()
            # Auto-cache is usually inside locales
            self.auto_cache_path = (self.locales_path / Path(
                auto_cache_dir).name).resolve()  # Ensure only dir name is used if full path given

            # *** Critical Debugging Logs ***
            logger.info(f"DEBUG: translate.py location: {Path(__file__).resolve()}")
            logger.info(f"DEBUG: Calculated project root: {self.project_root}")
            logger.info(f"DEBUG: Attempting to load locales from: {self.locales_path}")
            logger.info(f"DEBUG: Does locales path exist? {self.locales_path.exists()}")
            logger.info(f"DEBUG: Is locales path a directory? {self.locales_path.is_dir()}")
            logger.info(f"DEBUG: Attempting to load auto-cache from: {self.auto_cache_path}")
            # *** End Debugging Logs ***

        except Exception as e:
            logger.critical(f"CRITICAL: Error resolving paths: {e}. Translation might fail.", exc_info=True)
            # Fallback to relative paths - less reliable, depends on CWD
            self.locales_path = Path(locales_dir)
            self.auto_cache_path = Path(auto_cache_dir)

        self.default_lang = default_lang
        self.default_mode = default_mode
        self.translations = {}
        self.auto_translation_cache = {}
        self.supported_langs = []
        self.online_translator = None
        self.online_translator_initialized = False

        # Ensure directories exist
        try:
            self.locales_path.mkdir(parents=True, exist_ok=True)
            self.auto_cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directories {self.locales_path} or {self.auto_cache_path}: {e}")

        # Load translations and cache
        self._load_local_translations()
        self._load_auto_translation_cache()

        # --- Session State Initialization (Using zh-cn) ---
        session_lang = st.session_state.get('lang')
        # Set default lang if not set or not supported
        if not session_lang or session_lang not in self.supported_langs:
            st.session_state.lang = self.default_lang
            if session_lang:
                logger.warning(f"Session lang '{session_lang}' invalid, reset to '{self.default_lang}'")
            else:
                logger.info(f"Session lang initialized to '{self.default_lang}'")
        # Correct 'zh' to 'zh-cn' if found
        elif session_lang == 'zh':
            st.session_state.lang = 'zh-cn'
            logger.info("Corrected session language from 'zh' to 'zh-cn'")

        # Ensure default_lang itself is valid based on loaded files
        if self.default_lang not in self.supported_langs and self.supported_langs:
            self.default_lang = self.supported_langs[0]
            logger.warning(
                f"Default language file '{default_lang}' missing, changed default to '{self.supported_langs[0]}'")
        elif not self.supported_langs:
            logger.error("CRITICAL: No language files loaded. Translation system inactive.")
            self.default_lang = None  # No valid default

        # Initialize translation mode
        self.valid_modes = ['local_only', 'local_then_auto']
        session_mode = st.session_state.get('translation_mode')
        if not session_mode or session_mode not in self.valid_modes:
            st.session_state.translation_mode = self.default_mode
            if session_mode:
                logger.warning(f"Session mode '{session_mode}' invalid, reset to '{self.default_mode}'")
            else:
                logger.info(f"Session mode initialized to '{self.default_mode}'")

        logger.info(
            f"Hybrid Translator initialized. Supported langs: {self.supported_langs}, Default: {self.default_lang}, Mode: {st.session_state.translation_mode}")

    def _load_local_translations(self):
        """Load pre-defined translations from json files using resolved path."""
        self.translations = {}
        self.supported_langs = []
        logger.debug(f"Loading local translations from: {self.locales_path}")
        if not self.locales_path.is_dir():
            logger.error(f"Local translation directory does not exist or is not a directory: {self.locales_path}")
            return

        loaded_count = 0
        for file_path in self.locales_path.glob("*.json"):
            # Skip cache files if they are inside locales dir itself
            if file_path.name.startswith("auto_cache_"):
                continue

            lang_code = file_path.stem  # e.g., 'en', 'zh-cn'
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations[lang_code] = json.load(f)
                self.supported_langs.append(lang_code)
                logger.info(f"Loaded local translations for language: {lang_code} from {file_path}")
                loaded_count += 1
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse local JSON file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to read local translation file {file_path}: {e}")

        if loaded_count == 0:
            logger.warning(f"No valid local translation files (.json) found in {self.locales_path}.")
        else:
            logger.info(f"Loaded {loaded_count} local translation language(s): {self.supported_langs}")

    def _load_auto_translation_cache(self):
        """Load cached automatic translations from files using resolved path."""
        self.auto_translation_cache = {}
        logger.debug(f"Loading automatic translation cache from: {self.auto_cache_path}")
        if not self.auto_cache_path.is_dir():
            logger.warning(f"Auto-cache directory does not exist: {self.auto_cache_path}. Will be created on save.")
            return

        loaded_count = 0
        for file_path in self.auto_cache_path.glob("auto_cache_*.json"):
            try:
                lang_code = file_path.stem.replace("auto_cache_", "")
                # Load cache only if the language is supported by local files
                if lang_code in self.supported_langs:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.auto_translation_cache[lang_code] = json.load(f)
                    logger.info(f"Loaded automatic translation cache for language: {lang_code}")
                    loaded_count += 1
                else:
                    logger.debug(f"Skipping auto cache file for unsupported language: {file_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse auto-cache JSON file {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to read auto-cache file {file_path}: {e}")

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} automatic translation cache file(s).")

    def _save_auto_translation_cache(self, lang: str):
        """Save the automatic translation cache for a specific language to a file."""
        if lang not in self.auto_translation_cache or not self.auto_translation_cache[lang]:
            logger.debug(f"No auto-translation cache data to save for language: {lang}")
            return

        file_path = self.auto_cache_path / f"auto_cache_{lang}.json"
        try:
            # Ensure directory exists
            self.auto_cache_path.mkdir(parents=True, exist_ok=True)

            cache_to_save = self.auto_translation_cache[lang].copy()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_to_save, f, ensure_ascii=False, indent=2, sort_keys=True)
            logger.debug(f"Saved automatic translation cache for language: {lang} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save automatic translation cache for {lang} to {file_path}: {e}")

    # --- Temporarily disable online translator for debugging local files ---
    def _init_online_translator(self) -> bool:
        """Initializes the online translator instance (google_trans_new)."""
        logger.warning("Online translator is temporarily disabled for debugging local files.")
        self.online_translator_initialized = True  # Mark as attempted
        self.online_translator = None
        return False  # Force return False

    # --- End temporary disabling ---

    # --- Original _init_online_translator using google_trans_new (keep for later) ---
    # def _init_online_translator(self) -> bool:
    #     """Initializes the online translator instance (google_trans_new)."""
    #     if self.online_translator_initialized:
    #         return self.online_translator is not None

    #     self.online_translator_initialized = True # Mark as initialization attempted

    #     if not GOOGLETRANS_AVAILABLE:
    #         logger.warning("google_trans_new library not available.")
    #         return False

    #     if self.online_translator is None:
    #         logger.info("Initializing online translator (google_trans_new)...")
    #         try:
    #             self.online_translator = google_trans_new.google_translator(timeout=5) # Add a timeout
    #             test_dest_lang = self.default_lang if self.default_lang in self.supported_langs else 'en'
    #             if not test_dest_lang:
    #                  logger.error("Cannot perform online translator test: No valid destination language available.")
    #                  self.online_translator = None; return False

    #             test_result = self.online_translator.translate("test", lang_tgt=test_dest_lang)
    #             if isinstance(test_result, list): test_result = test_result[0]

    #             if isinstance(test_result, str):
    #                 logger.info(f"Online translator (google_trans_new) initialized successfully. Test result: '{test_result}'")
    #                 return True
    #             else:
    #                  logger.error(f"Online translator test returned unexpected type: {type(test_result)}. Result: {test_result}")
    #                  self.online_translator = None; return False
    #         except Exception as e:
    #             logger.error(f"Failed to initialize or test google_trans_new Translator: {e}", exc_info=True)
    #             self.online_translator = None; return False
    #     return self.online_translator is not None
    # --- End original online translator init ---

    def t(self, key: str, lang: Optional[str] = None, fallback: Optional[str] = None, **kwargs) -> str:
        """
        Gets the translation for a key using the current mode.
        """
        target_lang = lang if lang else st.session_state.get('lang', self.default_lang)
        mode = st.session_state.get('translation_mode', self.default_mode)

        # Validate target language
        if not target_lang or target_lang not in self.supported_langs:
            target_lang = self.default_lang
            if not target_lang:
                logger.error(f"Translation failed for key '{key}': No valid target or default language.")
                return fallback if fallback is not None else f"[{key}]"

        # --- 1. Try Local Translations ---
        if target_lang in self.translations:
            translated_text = self.translations[target_lang].get(key)
            if translated_text is not None:
                logger.debug(f"Local translation found: lang='{target_lang}', key='{key}'")
                try:
                    return translated_text.format(**kwargs) if kwargs else translated_text
                except Exception as fmt_e:
                    logger.warning(
                        f"Local format error (lang='{target_lang}', key='{key}'): {fmt_e}. Text: '{translated_text}'")
                    return translated_text

        # --- 2. Try Auto-Cache (if mode allows) ---
        if mode == 'local_then_auto':
            if target_lang in self.auto_translation_cache:
                cached_translation = self.auto_translation_cache[target_lang].get(key)
                if cached_translation is not None:
                    logger.debug(f"Auto-cache translation found: lang='{target_lang}', key='{key}'")
                    try:
                        return cached_translation.format(**kwargs) if kwargs else cached_translation
                    except Exception as fmt_e:
                        logger.warning(
                            f"Auto-cache format error (lang='{target_lang}', key='{key}'): {fmt_e}. Text: '{cached_translation}'")
                        return cached_translation

            # --- 3. Try Online Translation (if mode allows and not in cache) ---
            if self._init_online_translator() and self.online_translator:
                logger.debug(f"Performing online translation: lang='{target_lang}', key='{key}'")
                try:
                    source_text_for_api = key
                    if self.default_lang and self.default_lang in self.translations:
                        source_text_for_api = self.translations[self.default_lang].get(key, key)

                    online_result = self.online_translator.translate(source_text_for_api, lang_tgt=target_lang)
                    if isinstance(online_result, list): online_result = online_result[0]

                    if isinstance(online_result, str) and online_result.strip():
                        logger.info(
                            f"Online translation success: lang='{target_lang}', key='{key}' => '{online_result}'")
                        if target_lang not in self.auto_translation_cache: self.auto_translation_cache[target_lang] = {}
                        self.auto_translation_cache[target_lang][key] = online_result
                        self._save_auto_translation_cache(target_lang)
                        try:
                            return online_result.format(**kwargs) if kwargs else online_result
                        except Exception as fmt_e:
                            logger.warning(
                                f"Online translation format error (lang='{target_lang}', key='{key}'): {fmt_e}. Text: '{online_result}'")
                            return online_result
                    else:
                        logger.warning(
                            f"Online translation returned empty/invalid: key='{key}', lang='{target_lang}', result='{online_result}'")
                except Exception as e_translate:
                    logger.error(f"Online translation failed: key='{key}', lang='{target_lang}', error='{e_translate}'")
            else:
                logger.debug(
                    f"Online translation skipped: key='{key}', mode='{mode}', translator_ok={self.online_translator is not None}")

        # --- 4. Final Fallback ---
        final_fallback = fallback if fallback is not None else f"[{key}]"  # Return key in brackets if completely failed
        logger.warning(
            f"Translation failed for key '{key}' in lang '{target_lang}' (mode: {mode}). Returning fallback: '{final_fallback}'")
        return final_fallback

    def add_switcher(self, location=st.sidebar):
        """Adds language and mode switchers to the UI."""
        if not self.supported_langs: self._load_local_translations()

        with location:
            st.divider()
            st.markdown(f"**{self.t('ui_settings', fallback='界面设置')}**")

            # --- Language Switcher ---
            if self.supported_langs:
                current_lang = st.session_state.get('lang', self.default_lang)
                if current_lang not in self.supported_langs: current_lang = self.default_lang

                if current_lang:  # Ensure we have a valid current language
                    lang_display_names = {'en': 'English', 'zh-cn': '简体中文'}
                    lang_options_display = [lang_display_names.get(lang, lang) for lang in self.supported_langs]
                    try:
                        current_lang_index = self.supported_langs.index(current_lang)
                    except ValueError:  # Should not happen if logic above is correct
                        current_lang_index = 0

                    selected_lang_display = location.selectbox(
                        label=self.t('select_language_label', fallback="选择语言"),
                        options=lang_options_display, index=current_lang_index, key="language_selector"
                    )

                    selected_lang_code = next((code for code, display in lang_display_names.items() if
                                               display == selected_lang_display and code in self.supported_langs),
                                              selected_lang_display if selected_lang_display in self.supported_langs else current_lang)

                    if selected_lang_code != current_lang:
                        st.session_state.lang = selected_lang_code
                        logger.info(f"Language changed via UI to: {st.session_state.lang}")
                        st.rerun()
            else:
                location.caption(self.t('no_language_files_found', fallback="未找到语言文件。"))

            # --- Translation Mode Switcher ---
            current_mode = st.session_state.get('translation_mode', self.default_mode)
            mode_options = {
                'local_only': self.t('mode_local_only', fallback="仅本地翻译 (最快)"),
                'local_then_auto': self.t('mode_local_then_auto', fallback="本地优先+自动翻译缓存 (推荐)")
            }
            mode_keys = list(mode_options.keys())
            if current_mode not in mode_keys: current_mode = self.default_mode
            current_mode_index = mode_keys.index(current_mode)

            selected_mode_display = location.radio(
                label=self.t('translation_mode_label', fallback="翻译模式"),
                options=[mode_options[key] for key in mode_keys],
                index=current_mode_index, key="translation_mode_selector", horizontal=True,
            )
            selected_mode_key = next((key for key, display in mode_options.items() if display == selected_mode_display),
                                     current_mode)

            if selected_mode_key != current_mode:
                st.session_state.translation_mode = selected_mode_key
                logger.info(f"Translation mode changed to: {st.session_state.translation_mode}")

            # Display online translator status
            if st.session_state.translation_mode == 'local_then_auto':
                status_key = "checking"
                fallback_status = "检查中..."
                if self.online_translator_initialized:
                    status_key = "online_translator_available" if self.online_translator else "online_translator_unavailable"
                    fallback_status = "可用" if self.online_translator else "不可用/初始化失败"
                location.caption(
                    f"{self.t('online_translator_status', fallback='在线翻译状态')}: {self.t(status_key, fallback=fallback_status)}")

            st.divider()


# --- Global Instance Creation ---
translator: AutoTranslator  # Type hint for IDE

try:
    _current_script_dir = Path(__file__).parent
    # *** 根据您的项目结构调整这里 ***
    _project_root = _current_script_dir.parent.parent # 假设 core 在根目录下一级
    _locales_path = (_project_root / "locales").resolve()
    _auto_cache_path = (_locales_path / "auto_cache").resolve()

    # *** 添加/取消注释 调试日志 ***
    logger.critical(f"DEBUG: translate.py location: {Path(__file__).resolve()}")
    logger.critical(f"DEBUG: Calculated project root: {_project_root}")
    logger.critical(f"DEBUG: Attempting to load locales from: {_locales_path}")
    logger.critical(f"DEBUG: Does locales path exist? {_locales_path.exists()}")
    logger.critical(f"DEBUG: Is locales path a directory? {_locales_path.is_dir()}")
    # *** 结束调试日志 ***

    translator = AutoTranslator(locales_dir=_locales_path, auto_cache_dir=_auto_cache_path, default_lang="zh-cn")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize AutoTranslator: {e}", exc_info=True)


    # Fallback to a dummy translator if initialization fails catastrophically
    class FakeTranslator:
        def t(self, key: str, fallback: Optional[str] = None,
              **kwargs) -> str: return fallback if fallback else f"[{key}]"

        def add_switcher(self, location=st.sidebar): pass

        def _init_online_translator(self): return False


    translator = FakeTranslator()
    # Try to show error in Streamlit UI if possible, though st might not be fully ready here
    try:
        import streamlit as st_err

        st_err.error("FATAL: Translation system failed to initialize. UI text will be missing or show keys.")
    except ImportError:
        pass  # Cannot show error in UI if streamlit not available