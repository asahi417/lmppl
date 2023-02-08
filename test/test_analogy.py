tmp = """
Which one of the following is an analogy?
1) ''story'' is to ''building'' what ''crust'' is to ''sandwich''
2) ''story'' is to ''building'' what ''shingle'' is to ''roof''
3) ''story'' is to ''building'' what ''data'' is to ''file''
4) ''story'' is to ''building'' what ''layer'' is to ''cake''
5) ''story'' is to ''building'' what ''root'' is to ''plant''
The correct answer is
"""
# 4
from lmppl import EncoderDecoderLM
tmp = """
Which one of the following is an analogy?
1) architect is to blueprint what instructor is to blackboard
2) architect is to blueprint what graduate is to diploma
3) architect is to blueprint what musician is to note
4) architect is to blueprint what painter is to brush
5) architect is to blueprint what composer is to score
The correct answer is
"""
model = EncoderDecoderLM('google/flan-t5-base', max_length_decoder=32)
# model = EncoderDecoderLM('google/flan-t5-large', max_length_encoder=None, max_length_decoder=64)
# model = EncoderDecoderLM('t5-base', max_length_encoder=128, max_length_decoder=64)
# model = EncoderDecoderLM('google/flan-t5-xl', max_length_encoder=None, max_length_decoder=64)
# model = EncoderDecoderLM('google/t5-large', max_length_encoder=None, max_length_decoder=64)
score = model.get_perplexity(
    input_texts=[tmp] * 5,
    output_texts=['1)', '2)', '3)', '4)', '5)'],
)
print(['1)', '2)', '3)', '4)', '5)'][score.index(min(score))])  # 5


score = model.get_perplexity(
    input_texts=["''story'' is to ''building'' what"] * 5,
    output_texts=[
        "''instructor'' is to ''blackboard''",
        "''graduate'' is to ''diploma''",
        "''musician'' is to ''note''",
        "''painter'' is to ''brush''",
        "''composer'' is to ''score''"
    ]
)
print(['1)', '2)', '3)', '4)', '5)'][score.index(min(score))])  # 5
