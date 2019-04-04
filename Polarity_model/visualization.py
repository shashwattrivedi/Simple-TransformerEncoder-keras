import pandas as pd

class CharVal(object):
    def __init__(self, char, val):
        self.char = char
        self.val = val

    def __str__(self):
        return self.char

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
def color_charvals(s):
    r = 255-int(s.val*255)
    color = rgb_to_hex((255, r, r))
    return 'background-color: %s' % color


def display_attention(tokenized_text,attention_output):
	"""
	Simple Attention visualization using pandas styling

	Source : https://stackoverflow.com/questions/53867351/how-to-visualize-attention-weights

	"""
	char_vals = [CharVal(c, v) for c, v in zip(tokenized_text, attention_output)]

	char_df = pd.DataFrame(char_vals).transpose()
	# apply coloring values
	char_df = char_df.style.applymap(color_charvals)
	return char_df