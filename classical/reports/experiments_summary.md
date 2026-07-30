Окружение: Python 3.13.12, scikit-learn 1.8.0, NumPy 2.4.1, SciPy 1.17.1, pandas 3.0.0.

| этап | конфигурация | оценка на | F1 toxic | ± std | macro-F1 | precision | recall | AP |
|---|---|---|---|---|---|---|---|---|
| stage3_baselines | dummy_most_frequent | validation | 0.0000 |  | 0.4001 | 0.0000 | 0.0000 |  |
| stage3_baselines | dummy_stratified | validation | 0.3374 |  | 0.5041 | 0.3386 | 0.3362 | 0.3349 |
| stage3_baselines | word_1_tfidf_logreg | validation | 0.6997 |  | 0.7937 | 0.9016 | 0.5716 | 0.8577 |
| stage3_baselines | word_1_tfidf_logreg_cased | validation | 0.6743 |  | 0.7779 | 0.8965 | 0.5404 | 0.8486 |
| stage4_word_models | word_11_logreg_C1 | validation | 0.6997 |  | 0.7937 | 0.9016 | 0.5716 | 0.8577 |
| stage4_word_models | word_12_logreg_C1 | validation | 0.6661 |  | 0.7739 | 0.9156 | 0.5234 | 0.8522 |
| stage4_word_models | word_13_logreg_C1 | validation | 0.6545 |  | 0.7668 | 0.9158 | 0.5092 | 0.8510 |
| stage4_word_models | word_11_linearsvc_C1 | validation | 0.7717 |  | 0.8360 | 0.8525 | 0.7050 | 0.8679 |
| stage4_word_models | word_12_linearsvc_C1 | validation | 0.7756 |  | 0.8392 | 0.8641 | 0.7035 | 0.8637 |
| stage4_word_models | word_13_linearsvc_C1 | validation | 0.7747 |  | 0.8384 | 0.8599 | 0.7050 | 0.8634 |
| stage4_word_models | word_12_logreg_C4 | validation | 0.7549 |  | 0.8276 | 0.8902 | 0.6553 | 0.8636 |
| stage4_word_models | word_12_logreg_C16 | validation | 0.7770 |  | 0.8408 | 0.8741 | 0.6993 | 0.8629 |
| stage4_word_models | word_12_linearsvc_C0.3 | validation | 0.7492 |  | 0.8243 | 0.8955 | 0.6440 | 0.8667 |
| stage4_word_models | word_12_linearsvc_C3 | validation | 0.7658 |  | 0.8304 | 0.8284 | 0.7121 | 0.8535 |
| stage5_char_models | word12_logreg | validation | 0.7770 |  | 0.8408 | 0.8741 | 0.6993 | 0.8629 |
| stage5_char_models | word12_logreg_balanced | validation | 0.7953 |  | 0.8491 | 0.8237 | 0.7688 | 0.8624 |
| stage5_char_models | word12_linearsvc | validation | 0.7756 |  | 0.8392 | 0.8641 | 0.7035 | 0.8637 |
| stage5_char_models | word12_linearsvc_balanced | validation | 0.7868 |  | 0.8429 | 0.8168 | 0.7589 | 0.8613 |
| stage5_char_models | charwb25_logreg | validation | 0.8397 |  | 0.8839 | 0.9091 | 0.7801 | 0.9296 |
| stage5_char_models | charwb25_logreg_balanced | validation | 0.8507 |  | 0.8898 | 0.8790 | 0.8241 | 0.9296 |
| stage5_char_models | charwb25_linearsvc | validation | 0.8463 |  | 0.8883 | 0.9075 | 0.7929 | 0.9297 |
| stage5_char_models | charwb25_linearsvc_balanced | validation | 0.8521 |  | 0.8908 | 0.8805 | 0.8255 | 0.9290 |
| stage5_char_models | char15_logreg | validation | 0.8390 |  | 0.8836 | 0.9132 | 0.7759 | 0.9283 |
| stage5_char_models | char15_logreg_balanced | validation | 0.8592 |  | 0.8962 | 0.8892 | 0.8312 | 0.9283 |
| stage5_char_models | char15_linearsvc | validation | 0.8427 |  | 0.8861 | 0.9124 | 0.7830 | 0.9284 |
| stage5_char_models | char15_linearsvc_balanced | validation | 0.8546 |  | 0.8931 | 0.8908 | 0.8213 | 0.9281 |
| stage5_char_models | word12+charwb25_logreg | validation | 0.8373 |  | 0.8822 | 0.9073 | 0.7773 | 0.9232 |
| stage5_char_models | word12+charwb25_logreg_balanced | validation | 0.8402 |  | 0.8826 | 0.8779 | 0.8057 | 0.9232 |
| stage5_char_models | word12+charwb25_linearsvc | validation | 0.8314 |  | 0.8774 | 0.8900 | 0.7801 | 0.9210 |
| stage5_char_models | word12+charwb25_linearsvc_balanced | validation | 0.8426 |  | 0.8843 | 0.8796 | 0.8085 | 0.9214 |
| stage5_char_models | charwb25_linearsvc_balanced_cased | validation | 0.8590 |  | 0.8958 | 0.8855 | 0.8340 | 0.9306 |
| stage5_char_models | charwb25_linearsvc_balanced_no_punct | validation | 0.8542 |  | 0.8924 | 0.8833 | 0.8270 | 0.9256 |
| stage5_char_models | word12_linearsvc_balanced_punct_tokens | validation | 0.7796 |  | 0.8381 | 0.8145 | 0.7475 | 0.8627 |
| stage6_cv_selection | charwb25_linearsvc_C0.3_balanced | train 5-fold CV | 0.8686 | 0.0097 | 0.9015 | 0.8680 | 0.8693 |  |
| stage6_cv_selection | charwb25_cased_linearsvc_C0.3_balanced | train 5-fold CV | 0.8654 | 0.0104 | 0.8990 | 0.8636 | 0.8675 |  |
| stage6_cv_selection | charwb25_logreg_C4_balanced | train 5-fold CV | 0.8653 | 0.0101 | 0.8990 | 0.8655 | 0.8650 |  |
| stage6_cv_selection | charwb25_linearsvc_C1_balanced | train 5-fold CV | 0.8653 | 0.0099 | 0.8995 | 0.8735 | 0.8574 |  |
| stage6_cv_selection | charwb25_cased_linearsvc_C1_balanced | train 5-fold CV | 0.8643 | 0.0089 | 0.8989 | 0.8750 | 0.8541 |  |
| stage6_cv_selection | charwb25_logreg_C16_balanced | train 5-fold CV | 0.8627 | 0.0131 | 0.8975 | 0.8709 | 0.8547 |  |
| stage6_cv_selection | charwb25_cased_logreg_C4_balanced | train 5-fold CV | 0.8627 | 0.0112 | 0.8970 | 0.8609 | 0.8647 |  |
| stage6_cv_selection | charwb25_cased_logreg_C16_balanced | train 5-fold CV | 0.8608 | 0.0084 | 0.8963 | 0.8720 | 0.8502 |  |
| stage6_cv_selection | char15_linearsvc_C0.3_balanced | train 5-fold CV | 0.8608 | 0.0115 | 0.8960 | 0.8661 | 0.8556 |  |
| stage6_cv_selection | char15_linearsvc_C1_balanced | train 5-fold CV | 0.8601 | 0.0129 | 0.8962 | 0.8788 | 0.8422 |  |
| stage6_cv_selection | charwb25_cased_logreg_C64_balanced | train 5-fold CV | 0.8585 | 0.0065 | 0.8948 | 0.8739 | 0.8438 |  |
| stage6_cv_selection | char15_cased_linearsvc_C1_balanced | train 5-fold CV | 0.8581 | 0.0113 | 0.8949 | 0.8794 | 0.8380 |  |
| stage6_cv_selection | charwb25_logreg_C64_balanced | train 5-fold CV | 0.8577 | 0.0102 | 0.8941 | 0.8716 | 0.8444 |  |
| stage6_cv_selection | char15_logreg_C4_balanced | train 5-fold CV | 0.8573 | 0.0129 | 0.8934 | 0.8622 | 0.8526 |  |
| stage6_cv_selection | char15_logreg_C16_balanced | train 5-fold CV | 0.8570 | 0.0137 | 0.8937 | 0.8727 | 0.8419 |  |
| stage6_cv_selection | char15_logreg_C64_balanced | train 5-fold CV | 0.8570 | 0.0122 | 0.8941 | 0.8781 | 0.8371 |  |
| stage6_cv_selection | char15_cased_linearsvc_C0.3_balanced | train 5-fold CV | 0.8553 | 0.0138 | 0.8922 | 0.8649 | 0.8462 |  |
| stage6_cv_selection | charwb25_cased_linearsvc_C3_balanced | train 5-fold CV | 0.8550 | 0.0051 | 0.8926 | 0.8757 | 0.8356 |  |
| stage6_cv_selection | char15_cased_logreg_C64_balanced | train 5-fold CV | 0.8541 | 0.0122 | 0.8921 | 0.8778 | 0.8319 |  |
| stage6_cv_selection | char15_cased_logreg_C16_balanced | train 5-fold CV | 0.8536 | 0.0117 | 0.8914 | 0.8722 | 0.8359 |  |
| stage6_cv_selection | charwb25_linearsvc_C3_balanced | train 5-fold CV | 0.8534 | 0.0087 | 0.8913 | 0.8725 | 0.8353 |  |
| stage6_cv_selection | char15_cased_logreg_C4_balanced | train 5-fold CV | 0.8531 | 0.0122 | 0.8905 | 0.8632 | 0.8435 |  |
| stage6_cv_selection | char15_linearsvc_C3_balanced | train 5-fold CV | 0.8529 | 0.0106 | 0.8915 | 0.8824 | 0.8255 |  |
| stage6_cv_selection | char15_cased_linearsvc_C3_balanced | train 5-fold CV | 0.8516 | 0.0083 | 0.8906 | 0.8827 | 0.8228 |  |
| final | charwb25_linearsvc_C0.3_balanced (порог -0.0857) | validation | 0.8670 |  | 0.8991 | 0.8471 | 0.8879 | 0.9310 |
| final | charwb25_linearsvc_C0.3_balanced (порог -0.0857) | test | 0.8542 |  | 0.8892 | 0.8329 | 0.8766 | 0.9268 |
