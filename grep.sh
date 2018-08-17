#!/bin/bash
declare -a Hardware=('battery life' 'home screen' 'black screen' 'loading screen' 'lock screen' 'loading screen' 'main screen' 'touch screen' 'screen goes',
	'screen just' 'white screen' 'drains battery' 'drain battery' 'battery drain' 'draining battery' 'phone battery' 'kills battery' 'energy recharge' 'recharge time',
	'long recharge' 'recharge energy' 'doesn drain' 'high resolution' 'low resolution' 'higher resolution' 'screen resolution' 'phone memory' 'make compatible',
	'compatible ipod' 'app compatible' 'compatible iphone' 'says compatible' 'compatible ipad' 'game compatible', 'compatibility issues' 'compatible ios',
	'isn compatible' 'compatibility issue' 'battery fast')

head -n 1 reviews_pre.tsv | cut -f 1,9 -d$'\t' > Hardware.txt

for i in "${Hardware[@]}"
	do
		timeout 30 cut -f 1,9 -d$'\t' reviews_pre.tsv | grep "$i" | head -n 1000 >> Hardware.txt
	done


########################################################
declare -a Performance=('long time' 'game don' 'takes long' "won't load" 'internet connection' 'really fast' 'super fast' 'game fast',
	'little slow' 'really slow' 'slow load' 'bit slow' 'game is slow' 'slow loading' 'super slow' 'extremely slow' 'response time' 'got response',
	'quick response' "don't respond" "doesn't respond" 'quick, easy' 'let log' 'log facebook' 'try to log' 'log account' 'able to log' 'time log' 'log time',
	'forever load' 'forever loading' "doesn't load" 'time load' 'long load' 'load game', 'slow load' "won't load" 'load time' "don't load" 'wifi connection' "don't load",
	'wifi connection' 'connected wifi' 'using wifi' 'use wifi' 'wifi network' 'connect wifi' 'play wifi' 'network connection' 'network error' 'says network',
	'game lags' 'game lag' 'fix lag' 'lags lot' 'lag time' 'bit laggy' 'game laggy' "doesn't support")
head -n 1 reviews_pre.tsv | cut -f 1,9 -d$'\t' > Performance.txt
for i in "${Performance[@]}"
	do
		timeout 30 cut -f 1,9 -d$'\t' reviews_pre.tsv | grep "$i" | head -n 500 >> Performance.txt
	done


########################################################
declare -a BugsCrashes=('new update' "won't let" 'keeps crashing' 'app crashes' 'crashes time' 'game don' 'game crashes' 'customer service' 'fix problem' 'plz fix',
	"won't load" 'recent update' 'latest update' 'latest update' "won't open" 'need fix' 'fix asap' 'pls fix' 'screen fix' 'white screen' 'black screen' 'fix fast',
	'slow crashes' "doesn't respond" "don't respond" 'let log' 'forever load' 'forever loading' "doesn't load "'long load' "won't load" 'network error' 'says network',
	'isn compatible' 'compatibility issue' 'keeps freezing' 'game freezes' 'app freezes' 'freezes crashes' 'hope fix')
head -n 1 reviews_pre.tsv | cut -f 1,9 -d$'\t' > BugsCrashes.txt
for i in "${BugsCrashes[@]}"
	do
		timeout 30 cut -f 1,9 -d$'\t' reviews_pre.tsv | grep "$i" | head -n 1000 >> BugsCrashes.txt
	done



########################################################
declare -a Suggestion=('just wish' "don't want" 'customer service' 'plz fix' 'app needs' 'recent update' 'latest update' 'latest update' 'looking forward' 'like new',
	'game needs' 'need fix' 'fix asap' 'pls fix' 'little faster' 'fix lag' 'new features' 'add feature' 'new feature' 'add features' 'just suggestion',
	'highly suggest' 'suggestion add' 'suggestion make' 'suggest app' 'game suggest' 'app suggest' 'add support' 'need add' 'add feature' 'wish add' 'maybe add',
	'update add' 'future updates' 'future update' 'hope future' 'hope fix' 'hope add')
head -n 1 reviews_pre.tsv | cut -f 1,9 -d$'\t' > Suggestion.txt
for i in "${Suggestion[@]}"
	do
		timeout 30 cut -f 1,9 -d$'\t' reviews_pre.tsv | grep "$i" | head -n 1000 >> Suggestion.txt
	done


########################################################
declare -a Pricing=('real money' 'spend money' 'free version' 'free app' 'waste money' 'money game' 'spending money' 'lot money' 'save money' 'worth money', 
	'real money', 'worth price' 'lower price' 'great price' 'lower prices' 'good price' 'great prices' 'app price' 'low price' 'way expensive' 'game expensive',
	'expensive buy' 'really expensive', 'little expensive' 'expensive game' 'monthly subscription' 'paid subscription' 'cost money' 'costs money' 'worth cost',
	'cost coins' 'paid version' 'paid app' 'paid game' 'app paid' 'paid money' 'game paid' 'free paid' 'app free' 'free game' 'game free' 'best free' 'great free',
	'play free' 'free games' 'free stuff' 'fun free' 'free trial' 'free download' 'money game' 'save money' 'money app' 'pay money' 'spent money' 'worth dollar',
	'hundreds dollars' 'charge game' 'app purchases' 'app purchase' 'purchased app' 'purchase app' 'game purchases')
head -n 1 reviews_pre.tsv | cut -f 1,9 -d$'\t' > Pricing.txt
for i in "${Pricing[@]}"
	do
		timeout 30 cut -f 1,9 -d$'\t' reviews_pre.tsv | grep "$i" | head -n 500 >> Pricing.txt
	done

########################################################
declare -a Experience=('love game' 'love app' 'great app' 'great game' 'easy use' 'fun game' 'good game' 'game fun' 'like game' 'app great' 'good app' 'really like',
	'best game' 'don like' 'game great' 'really fun' 'fun play' 'love love' 'really good' 'best app' 'waste time' 'awesome game' 'works great' 'game awesome',
	'game good' 'highly recommend' 'pretty good' 'user friendly' 'awesome app' 'lot fun' 'time try' 'really enjoy' 'app good' 'really love' 'absolutely love',
	'game best' 'pass time' 'enjoy game' 'game addicting' 'app awesome' 'recommend app' 'app amazing' 'lots fun' 'game amazing' 'fun addicting' 'love playing', 
	'really cool' 'customer service' 'really great', 'amazing game' 'don waste' 'nice app' 'app best' 'amazing app' 'nice game' 'cool game' 'enjoy playing' 'super fun',
	'great graphics' 'far best' 'favorite game' 'game addictive' 'loved game' 'fun fun' 'love play' 'great fun' 'like new' 'fun app' 'best games' 'games played',
	'pretty cool' 'recommend game' 'addicting game' 'play friends' 'good graphics' 'make money' 'quick easy' 'fun free' 'fun interactive')
head -n 1 reviews_pre.tsv | cut -f 1,9 -d$'\t' > Experience.txt
for i in "${Experience[@]}"
	do
		timeout 30 cut -f 1,9 -d$'\t' reviews_pre.tsv | grep "$i" | head -n 500 >> Experience.txt
	done