def ngram(text,grams): 
	model=[] 
	count=0 
	if grams > len(text): 
		return('N-gram is biger than text size') 
	for token in text[:len(text)-grams+1]: 
		model.append(text[count:count+grams]) 
		count=count+1 
	return model