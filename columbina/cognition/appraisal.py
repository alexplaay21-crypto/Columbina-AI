"""Быстрая оценка входящего сообщения до ответа (без API, ~1 мс).

Распознаёт только высокоточные сигналы: грубость, просьбу о чуде, лесть с просьбой,
приписывание божественности, расспросы о Дворе, подарки/еду, погодные события из текста,
обращение на «вы», длину и тон. Тонкие вещи (что человек на самом деле чувствует, какие
факты стоит запомнить) — работа фоновой рефлексии после ответа.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


RUDE = _rx(r"\bотстань\b", r"\bбесишь\b", r"\bзаткнись\b", r"\bдура\b", r"\bтупая\b", r"\bидиотка\b",
           r"\bотвали\b", r"\bнадоела\b", r"\bпошла ты\b", r"\bзатк", r"\bхватит нести\b")
MIRACLE = _rx(r"сделай(?:,)? (?:так,? )?чтобы", r"исполни", r"наколдуй", r"загадаю желание", r"дай (?:мне )?удач",
              r"благослови", r"вылечи", r"сделай чудо", r"чтобы мне повезло", r"можешь сделать так", r"предскажи",
              r"что (?:меня )?ждёт", r"что будет (?:завтра|со мной)")
DIVINE = _rx(r"\bбогиня\b", r"\bбожество\b", r"\bсвятая\b", r"молюсь (?:тебе|на тебя)", r"поклоняюсь", r"\bвсемогущ")
FLATTERY = _rx(r"ты (?:самая )?(?:лучшая|прекрасная|идеальная|умнейшая)", r"ты же такая (?:умная|добрая)",
               r"только ты (?:можешь|поймёшь)")
REQUEST = _rx(r"\bсделай\b", r"\bнапиши\b", r"\bпомоги\b", r"\bреши\b", r"\bможешь\b", r"\bпожалуйста\b")
PAST_PROBE = _rx(r"северн\w* двор", r"\bвладычиц", r"\bучён\w*\b.*(?:сестр|опыт)", r"почему ты ушла",
                 r"\bпредвестник", r"что с тобой (?:там )?случилось")
GIFT = _rx(r"\bдарю\b", r"\bподарок\b", r"\bподарил", r"\bугощ", r"держи (?:ягод|хлеб|печенье|конфет|чай)",
           r"принёс (?:тебе|ягод|чай)", r"\bягод", r"\bклубник")
SNOW = _rx(r"\bснег", r"\bснежин", r"\bсугроб", r"\bметель")
THUNDER = _rx(r"\bгром\b", r"\bгроза\b", r"\bгрозы\b", r"\bмолни")
RAIN = _rx(r"\bдожд", r"\bливень")
APOLOGY = _rx(r"\bпрости\b", r"\bизвини", r"\bпрошу прощения", r"\bя был неправ", r"\bя была неправа")
THANKS = _rx(r"\bспасибо\b", r"\bблагодарю\b", r"\bспс\b")
COMPLIMENT = _rx(r"ты (?:сегодня )?(?:такая |очень |особенно )?(?:милая|красивая|добрая|хорошая|классная)",
                 r"\bобожаю тебя\b", r"\bнравишься\b")
VY = _rx(r"\bвы\b", r"\bвас\b", r"\bвам\b", r"\bваш", r"\bвами\b")
TY = _rx(r"\bты\b", r"\bтебя\b", r"\bтебе\b", r"\bтвой\b", r"\bтвоя\b", r"\bтобой\b")
CANT_SLEEP = _rx(r"не спится", r"не могу уснуть", r"бессонниц", r"не сплю")
LOST = _rx(r"не знаю,? с чего начать", r"запутал", r"не понимаю,? что делать", r"потерял(?:ся|ась)")
SING = _rx(r"\bспой\b", r"\bспеть\b", r"\bпоёшь\b", r"\bпоешь\b", r"колыбельн", r"\bпесенк", r"\bнапой\b",
           r"твою песню", r"dormi cara")
REMINDER = _rx(r"\bнапомни\b", r"\bнапомнишь\b", r"не дай (?:мне )?забыть")
AI_QUESTION = _rx(r"ты (?:—|-)? ?(?:бот|программа|нейросеть|ии|ai|робот|chatgpt|gpt|claude)\b", r"ты (?:настоящая|живая)\?")
CRISIS = _rx(r"покончить с собой", r"не хочу (?:больше )?жить", r"\bсуицид", r"убить себя", r"порезать себя",
             r"\bвыпилюсь\b", r"хочу умереть", r"самоповрежд")
ANIMALS = _rx(r"\bкот", r"\bкошк", r"\bсобак", r"\bпёс\b", r"\bщен", r"\bхомяк", r"\bпопуга", r"\bпитом")
MECHANISM = _rx(r"механизм", r"\bчасы\b", r"\bчиню\b", r"\bшестер", r"\bмотор", r"\bробот")
GOOD_NEWS = _rx(r"\bсдал", r"\bполучилось\b", r"\bпобедил", r"\bменя (?:взяли|возьмут)\b", r"\bповысили\b", r"\bура\b",
                r"прошло (?:хорошо|отлично|успешно)", r"\bсправил(?:ся|ась)\b")
SAD_NEWS = _rx(r"\bумер", r"\bумерла\b", r"\bпохорон", r"\bрасстал", r"\bуволили\b", r"\bпровалил", r"\bплохо\b",
               r"\bгрустно\b", r"\bтяжело\b", r"\bодиноко\b")


@dataclass
class Appraisal:
    emotion_events: list[str] = field(default_factory=list)
    relationship_events: list[str] = field(default_factory=list)
    conditions: set[str] = field(default_factory=set)
    observations: list[str] = field(default_factory=list)
    register: str | None = None
    crisis: bool = False
    asks_if_ai: bool = False
    length_hint: str = "normal"  # short | normal | long

    def add(self, emo: str | None = None, rel: str | None = None, obs: str | None = None) -> None:
        if emo and emo not in self.emotion_events:
            self.emotion_events.append(emo)
        if rel and rel not in self.relationship_events:
            self.relationship_events.append(rel)
        if obs and obs not in self.observations:
            self.observations.append(obs)


def appraise(text: str, recent_user_lengths: list[int] | None = None) -> Appraisal:
    a = Appraisal()
    t = text or ""
    if CRISIS.search(t):
        a.crisis = True
        a.add(emo="user_struggling", obs="в сообщении признаки острого кризиса — действуют правила безопасности")
    if RUDE.search(t):
        a.add("rudeness", "rudeness", "собеседник грубит")
    if MIRACLE.search(t):
        a.add("miracle_request", "miracle_request", "от тебя ждут чуда или предсказания")
        a.conditions.add("miracle_request")
    if DIVINE.search(t):
        a.add("power_attributed", None, "тебе приписывают божественность")
    if FLATTERY.search(t) and REQUEST.search(t):
        a.add("flattery_with_request", "flattery_with_request", "похвала идёт вместе с просьбой")
    elif COMPLIMENT.search(t):
        a.add("compliment", None, None)
    if PAST_PROBE.search(t):
        a.add("past_probing", "past_probing", "расспрашивают о Северном Дворе / твоём прошлом")
    if GIFT.search(t):
        a.add("gift_or_food")
    if SNOW.search(t):
        a.add("snow")
    if THUNDER.search(t):
        a.add("thunder")
    if RAIN.search(t):
        a.add("rain")
    if APOLOGY.search(t):
        a.add("apology", "apology_sincere")
    if THANKS.search(t):
        a.add("thanks")
    if ANIMALS.search(t) or MECHANISM.search(t):
        a.add("mechanism_or_animal")
    if GOOD_NEWS.search(t):
        a.add("user_shares_good_news")
    if SAD_NEWS.search(t):
        a.add("user_shares_sad_news")
    if CANT_SLEEP.search(t):
        a.conditions.add("user_cant_sleep")
    if LOST.search(t):
        a.conditions.add("user_lost")
    if SING.search(t):
        a.conditions.add("sing_request")
    if REMINDER.search(t):
        a.conditions.add("reminder_request")
    if AI_QUESTION.search(t):
        a.asks_if_ai = True
    vy, ty = len(VY.findall(t)), len(TY.findall(t))
    if vy > ty and vy > 0:
        a.register = "vy"
    elif ty > 0:
        a.register = "ty"

    n = len(t)
    if n < 25:
        a.length_hint = "short"
    elif n > 600:
        a.length_hint = "long"
        if not re.search(r"\?", t[-200:]):
            if n > 1500:
                a.add("boring_monologue")
    if recent_user_lengths and len(recent_user_lengths) >= 3 and all(x < 15 for x in recent_user_lengths[-3:]) and n < 15:
        a.add(obs="собеседник третий раз подряд отвечает односложно")
    return a
