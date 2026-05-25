代码仓应包含完整的示例数据集和详细文档，做到即插即用。后端同事只需修改输入数据路径、题目的内容、评分细则和总分，即可在新的数据上运行，无需改动核心代码。若需要调整算法训练或推断参数，只需通过配置文件或命令行参数即可实现。整个仓库应提供清晰的目录结构、可运行的 demo，以及参数化功能备注。

# 新增内容
1. 加入 sample 好的测试和训练数据集来 demo，将整个算法包做到即插即用  
2. 加入 baseline 方案（不估算 tier，只做 OCR 和打分）  
3. train.py 中默认训练数据没有做 OCR，加入 OCR 识别功能  

# 算法改进
1. current_rubric 从 dataset.json 中的 question 和 rubric 中读取（现在只处理一道题目，但实际需要处理新的题目，评分标准也可能变化，必须参数化；如需改良加入 Tier 信息，调用 LLM 执行优化）  
2. TIER_SYSTEM_PROMPT 中的 TIER 档位数参数化  
3. get_essay_tier 中若遇到大模型返回格式不对，能否重新循环调用 LLM 并加上提示确保返回 JSON 格式？exception 中基于字数打分和读取老师打分（实际上也不会有），return 4 对学生不公平  
4. prompts 里两个 EXTRACTION_SYSTEM_PROMPT，到底哪个？  
5. prompts 里的 REQUIRED_POINTS_JSON 从 question 和 rubric 中提取  
6. ASRO 和 autoscore_grade 中的 robust_extract_json 若遇到大模型返回格式不对，能否循环调用 LLM 并加提示确保返回 JSON 格式？有 JSON 读取 error 不能只给空值或 0 值，要克服格式问题取得真值  
7. prompts 里的 SCORING_USER_TEMPLATE 的 Tier 4 vs Tier 5 参数化  
8. data_loader.py 的 _get_tier 区分点设为总分的 50% 和 75%，每题总分作为参数从 args 传入  
9. train.py 里的 train_n 和 val_n 不需要预设，从 dataset_name 中的数据读取并 split  
10. train.py 里的 initial_G 中 prompts.TASK_6x_XXX 从 dataset.json 的 question 和 rubric 中读取  
11. ASRO/prompts 里的 GRADER_PROMPT_TEMPLATE 的 SCORE 和 TIER 范围参数化  
12. ASRO/utils/sampler.py 中 get_ordinal_score 返回的 lp_pred 和 lp_true 都是 -1，m_val 都是 0，seeds 并没有抽取 misconf 前 5？  
13. ASRO/utils/utils.py 中的 _score_to_tier 的 TIER 档位数参数化  
14. AES/ASRO/engine.py 中 _get_top_k_modes 的 weighted_cm 尺寸 31 参数化  
15. AES/ASRO/optimizer.py 中 _extract_json 若遇到大模型返回格式不对，能否循环调用 LLM 并加提示确保返回 JSON 格式？有 JSON 读取 error 不能只 pass，要克服格式问题取得真值  
16. AES/ASRO/engine.py 中 evaluate_minibatch_sequential 的 misconf 参数 10，以及 evaluate_validation_sequential 中的 misconf_val 按照学生答题值变化（专利讨论）  

# 工程改进
1. 添加只做 OCR 的参数  
2. OCR 调用多线程  
3. 所有 LLM 的 API 调用用 while True 无限循环方式执行  
4. LLM_API 及多线程调用参考 gpt4o_mini_inference/gpt4o_mini_inference_multithread 方案，通用函数化  
5. 由某个多线程的 bool 参数决定是否执行多线程  
6. 多线程尽量统一预处理完所有内容再并发调用 API  
7. prompts.py 加入分割和注释，说明不同算法下调用的 prompt 及功能  
8. AES/ASRO/engine.py 的 evaluate_minibatch_sequential 做到多线程  
9. train.py 中加入 debug 参数，debug 模式下 AES/ASRO/optimizer.py 执行 debug 算法及 _save_debug_log  
10. AES/ASRO/utils/data_loader.py 中若 D_val 拿不到数据怎么办？训练时无法判断  

# Bug 问题
1. 多余的参数 error_threshold  
2. process_single_essay 中读取 parts 不能处理文件名 `301011344_66.tif`，报错不返回 None 而是直接报错，确保用户执行有错必改  
3. AutoScoreClient 的 call_llm 中 payload 有多余 static 参数导致不能 call llm  
4. train.py 中加入 dataset_name args 并传给 GradeOptDataLoader  

# Readme
1. 简单讲解整体框架：三种方案可选 —— baseline（prompt 提取文本/打分）、autoscore、asro 训练 prompt，分别在不同场景使用  
2. inference 过程解释：先 OCR 预处理，再 ASRO/AutoScore  
3. 说明 ASRO 训练结果保存  
4. GitHub 重做 Composition + Filling 架构，Filling 先不传，并清掉过去 commit 历史  

# 其他问题
1. AutoScore 的 prompt SCORING_USER_TEMPLATE 和 CARO 的 prompt PURE_TEMPLATE 架构差不多？哪里需要手动调试作用大？  
2. 有没有可以封装 Python 脚本或部分脚本的方法  
