package qfirst.clause.ext

import jjm.LowerCaseString
import jjm.ling.en.InflectedForms
import jjm.implicits._

import cats.data.NonEmptyList
import cats.data.StateT
import cats.implicits._

import monocle.macros._

object TemplateStateMachine {

  @Lenses case class FrameState(
    whWord: Option[LowerCaseString],
    answerSlot: Option[ArgumentSlot],
    frame: Frame
  )

  object FrameState {
    def initial(verbForms: InflectedForms) = FrameState(None, None, Frame.empty(verbForms))
  }

  sealed trait TemplateState
  case object TemplateComplete extends TemplateState
  case class TemplateProgress(
    transitions: NonEmptyList[TemplateTransition]
  ) extends TemplateState {
    def +(transition: TemplateTransition) = TemplateProgress(
      transition :: transitions
    )
  }

  type TemplateTransition = (String, StateT[Option, FrameState, TemplateState])

  def pure[A](a: A) = StateT.pure[Option, FrameState, A](a)
  def abort[A] = StateT.liftF[Option, FrameState, A](None)
  def guard(cond: Boolean) = if(cond) pure(()) else abort[Unit]
  def get = StateT.get[Option, FrameState]
  def set(fs: FrameState) = StateT.set[Option, FrameState](fs)
  def modify(f: FrameState => FrameState) = StateT.modify[Option, FrameState](f)
  def modFrame(f: Frame => Frame) = StateT.modify[Option, FrameState](FrameState.frame.modify(f))
  def lift[A](aOpt: Option[A]) = StateT.liftF[Option, FrameState, A](aOpt)

  def modifyOpt(f: FrameState => Option[FrameState]) = for {
    fs <- get
    newFrameState <- lift(f(fs))
    _ <- set(newFrameState)
  } yield newFrameState

  def modifyFrameOpt(f: Frame => Option[Frame]) = for {
    fs <- get
    newFrame <- lift(f(fs.frame))
    _ <- set(fs.copy(frame = newFrame))
  } yield newFrame

  def progress(first: TemplateTransition, rest: TemplateTransition*) = TemplateProgress(
    NonEmptyList.of(first, rest: _*)
  )

  def markPlaceholderSlot[A](slot: ArgumentSlot.Aux[A], arg: A) =
    modifyFrameOpt(f => f.args.get(slot).ifEmpty(Frame.args.modify(_.put(slot, arg))(f)))

  def markAnswerSlot[A](
    slot: ArgumentSlot.Aux[A],
    makeArg: LowerCaseString => StateT[Option, FrameState, A]
  ) =
    for {
      fs     <- get
      _      <- lift(fs.answerSlot.ifEmpty(())) // only works if we don't already have an answer
      whWord <- lift(fs.whWord)
      arg    <- makeArg(whWord)
      newFrame <- lift(
        fs.frame.args
          .get(slot)
          .ifEmpty(
            Frame.args.modify(_.put(slot, arg))(fs.frame)
          )
      )
      _ <- set(fs.copy(frame = newFrame, answerSlot = Some(slot)))
    } yield ()

  // NOTE: is this vv still true?
  // neither of these should contain "to", which is handled specially

  val mostCommonPrepositions = Set(
    "by", "for", "with",
    // "about", // too many spurious questions from this
    "in", "from", "to", "as"
  ).map(_.lowerCase)

  val lotsOfPrepositions = Set(
    "aboard", "about", "above", "across", "afore", "after", "against", "ahead", "along", "alongside",
    "amid", "amidst", "among", "amongst", "around", "as", "aside", "astride", "at", "atop",
    "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "by", "despite",
    "down", "during", "except", "for", "from", "given", "in", "inside", "into", "near",
    "next", "of", "off", "on", "onto", "opposite", "out", "outside", "over", "pace",
    "per", "round", "since", "than", "through", "throughout", "till", "times", "to", "toward",
    "towards", "under", "underneath", "until", "unto", "up", "upon", "versus", "via", "with",
    "within", "without"
  ).map(_.lowerCase)

  val allPrepositions = mostCommonPrepositions ++ lotsOfPrepositions
}

class TemplateStateMachine(
  tokens: Vector[String],
  verbInflectedForms: InflectedForms,
  overridePrepositions: Option[Set[LowerCaseString]] = None
) {

  import TemplateStateMachine._

  val initialFrameState = FrameState.initial(verbInflectedForms)

  // process prepositions
  val lowerTokens = tokens.map(_.lowerCase)

  def isPreposition(lcs: LowerCaseString): Boolean =
    TemplateStateMachine.lotsOfPrepositions.contains(lcs)

  val detectedPrepositions = lowerTokens.filter(isPreposition).toSet

  val chosenPrepositions: NonEmptyList[LowerCaseString] =
    NonEmptyList(
      TemplateStateMachine.mostCommonPrepositions.head,
      (TemplateStateMachine.mostCommonPrepositions ++
          overridePrepositions.getOrElse(detectedPrepositions)).toList
    ).distinct

  import verbInflectedForms._

  def isFinalStateValid(fs: FrameState) = {
    fs.whWord.nonEmpty && fs.answerSlot.nonEmpty
  }

  val validateState = for {
    fs <- get
    _ <- guard(isFinalStateValid(fs))
  } yield ()

  val qMark = progress("?" -> validateState.as(TemplateComplete))

  def makeMiscForObjLocWh(wh: LowerCaseString): StateT[Option, FrameState, NonPrepArgument] =
    lift(
      if (wh == "who".lowerCase) Some(Noun(true))
      else if (wh == "what".lowerCase) Some(Noun(false))
      else if (wh == "where".lowerCase) Some(Locative)
      else None
    )

  def makeMiscForLocWh(wh: LowerCaseString): StateT[Option, FrameState, NonPrepArgument] =
    lift(
      if (wh == "where".lowerCase) Some(Locative)
      else None
    )

  def objLocMisc(hasObject: Boolean) = {
    if(hasObject) progress(
      ""           -> markAnswerSlot(Misc, makeMiscForObjLocWh).as(qMark),
      " someone"   -> markPlaceholderSlot(Misc, Noun(isAnimate = true)).as(qMark),
      " something" -> markPlaceholderSlot(Misc, Noun(isAnimate = false)).as(qMark),
      " somewhere" -> markPlaceholderSlot(Misc, Locative).as(qMark)
    ) else progress(
      ""           -> markAnswerSlot(Misc, makeMiscForLocWh).as(qMark),
      " somewhere" -> markPlaceholderSlot(Misc, Locative).as(qMark)
    )
  }

  def makeMiscForCompFormWh(form: Complement.Form) = (wh: LowerCaseString) => {
    lift(
      if(wh == "what".lowerCase) Some(Complement(form))
      else None
    ): StateT[Option, FrameState, NonPrepArgument]
  }

  def compObj(form: Complement.Form) = progress(
    "" -> markAnswerSlot(Misc, makeMiscForCompFormWh(form)).as(qMark),
    " something" -> markPlaceholderSlot(Misc, Complement(form)).as(qMark)
  )

  val makeGerundAnswer = (wh: LowerCaseString) => {
    lift(
      if(wh == "what".lowerCase) Some(Gerund)
      else None
    ): StateT[Option, FrameState, Gerund.type]
  }

  val gerundMiscObj = progress(
    "" -> markAnswerSlot(Misc, makeGerundAnswer.map(_.widen[NonPrepArgument])).as(qMark),
    " something" -> markPlaceholderSlot(Misc, Gerund).as(qMark)
  )

  def complement(followsPrep: Boolean) = {
    if(followsPrep) progress(
      " to do" -> pure(compObj(Complement.Form.Infinitive)),
      " doing" -> pure(gerundMiscObj)
    ) else progress(
      " to do" -> pure(compObj(Complement.Form.Infinitive)),
      // NOTE: this bare "do" with no preceding object is very often ungrammatical,
      // but it seems like it can work in rare cases like "What helps do something?" or "What does something help do?"
      " do" -> pure(compObj(Complement.Form.Bare)),
      " doing" -> pure(gerundMiscObj)
    )
  }

  def postObjMisc(hasObject: Boolean) = progress(
    "" -> pure(objLocMisc(hasObject)),
    "" -> pure(complement(followsPrep = false)),
    "" -> pure(qMark)
  )

  val postPrepMisc = progress(
    "" -> pure(complement(followsPrep = true)),
    "" -> pure(qMark)
  )

  def makePrepForWhObject(
    preposition: LowerCaseString
  )(wh: LowerCaseString): StateT[Option, FrameState, Preposition] =
    if (wh == "who".lowerCase) pure(Preposition(preposition, Some(Noun(true))))
    else if (wh == "what".lowerCase) pure(Preposition(preposition, Some(Noun(false))))
    else abort[Preposition]

  def makeGerundPrepObjAnswer(preposition: LowerCaseString) = (wh: LowerCaseString) => {
    makeGerundAnswer(wh).map(g => Preposition(preposition, Some(g)))
  }

  def prepGerundObj(
    prep: LowerCaseString,
    slot: ArgumentSlot.Aux[Preposition],
    next: TemplateState
  ) = progress(
    "" -> markAnswerSlot(slot, makeGerundPrepObjAnswer(prep)).as(next),
    " something" -> markPlaceholderSlot(slot, Preposition(prep, Some(Gerund))).as(next)
  )

  def prepObj(
    preposition: LowerCaseString,
    slot: ArgumentSlot.Aux[Preposition],
    next: TemplateState
  ) = progress(
    // prep has no object
    "" -> markPlaceholderSlot(slot, Preposition(preposition, None)).as(next),
    // asking about obj of prep
    "" -> markAnswerSlot(slot, makePrepForWhObject(preposition)).as(next),
    // prep has animate placeholder obj
    " someone" -> markPlaceholderSlot(slot, Preposition(preposition, Some(Noun(true)))).as(next),
    // prep has inanimate placeholder obj
    " something" -> markPlaceholderSlot(slot, Preposition(preposition, Some(Noun(false)))).as(next),
    // prep has gerund obj
    " doing" -> pure(prepGerundObj(preposition, slot, next))
  )

  def prep(
    slot: ArgumentSlot.Aux[Preposition],
    next: TemplateState
  ) = TemplateProgress(
    chosenPrepositions
      .map(prep => (" " + prep) -> pure(prepObj(prep, slot, next)))
  )

  def postObj(hasObject: Boolean) = progress(
    "" -> pure(
      prep(
        Prep1, prep(
          Prep2, postPrepMisc
        ) + ("" -> pure(postPrepMisc))
      )
    ),
    "" -> pure(postObjMisc(hasObject))
  )

  def makeNounForWh(wh: LowerCaseString): StateT[Option, FrameState, Noun] =
    lift(
      if (wh == "who".lowerCase) Some(Noun(true))
      else if (wh == "what".lowerCase) Some(Noun(false))
      else None
    )

  val obj = progress(
    ""           -> pure(postObj(false)),
    ""           -> markAnswerSlot(Obj, makeNounForWh).as(postObj(true)),
    " someone"   -> markPlaceholderSlot(Obj, Noun(true)).as(postObj(true)),
    " something" -> markPlaceholderSlot(Obj, Noun(false)).as(postObj(true))
  )

  // follows a have-aux. assume already isPerfect
  val pastParticipleVerb = progress(
    s" been $presentParticiple" -> modify(
      (FrameState.frame composeLens Frame.isProgressive).set(true)
    ).as(obj),
    s" been $pastParticiple" -> modify((FrameState.frame composeLens Frame.isPassive).set(true))
      .as(obj),
    (" " + pastParticiple.toString) -> pure(obj)
  )

  // follows a modal
  val infinitiveVerb = progress(
    (" " + stem.toString) -> pure(obj),
    s" be $presentParticiple" -> modify(
      (FrameState.frame composeLens Frame.isProgressive).set(true)
    ).as(obj),
    s" have been $presentParticiple" -> modify(
      (FrameState.frame composeLens Frame.isPerfect).set(true)
      andThen (FrameState.frame composeLens Frame.isProgressive).set(true)
    ).as(obj),
    s" be $pastParticiple" -> modify((FrameState.frame composeLens Frame.isPassive).set(true))
      .as(obj),
    s" have $pastParticiple" -> modify((FrameState.frame composeLens Frame.isPerfect).set(true))
      .as(obj),
    s" have been $pastParticiple" -> modify(
      (FrameState.frame composeLens Frame.isPerfect).set(true)
      andThen (FrameState.frame composeLens Frame.isPassive).set(true)
    ).as(obj)
  )

  // follows a do-aux
  val stemVerb = progress(
    " " + stem.toString -> pure(obj)
  )

  // follows a be-aux
  val presentParticipleOrPassiveVerb = progress(
    (" " + presentParticiple.toString) -> modify(
      (FrameState.frame composeLens Frame.isProgressive).set(true)
    ).as(obj),
    (s" being $pastParticiple") -> modify(
      (FrameState.frame composeLens Frame.isProgressive).set(true)
      andThen (FrameState.frame composeLens Frame.isPassive).set(true)
    ).as(obj),
    (" " + pastParticiple.toString) -> modify(
      (FrameState.frame composeLens Frame.isPassive).set(true)
    ).as(obj)
  )

  // follows no aux
  val tensedVerb = progress(
    (" " + presentSingular3rd.toString) -> modFrame(Frame.tense.set(Tense.Finite.Present)).as(obj),
    (" " + past.toString)    -> modFrame(Frame.tense.set(Tense.Finite.Past)).as(obj)
  )

  // neg/subj states carry the verb form through; so, all need to be constructed at construction time

  def postSubjectNegation(targetVerbState: TemplateState) = progress(
    ""     -> pure(targetVerbState),
    " not" -> modify((FrameState.frame composeLens Frame.isNegated).set(true)).as(targetVerbState)
  )

  def subj(targetVerbState: TemplateState, alreadyNegated: Boolean) = {
    val target = if (alreadyNegated) targetVerbState else postSubjectNegation(targetVerbState)
    progress(
      " someone"   -> markPlaceholderSlot(Subj, Noun(true)).as(target),
      " something" -> markPlaceholderSlot(Subj, Noun(false)).as(target),
      " it"        -> markPlaceholderSlot(Subj, Noun(false)).as(target)
    )
  }

  def optionalSubj(targetVerbState: TemplateState, alreadyNegated: Boolean) = {
    val skipSubjTarget =
      if (alreadyNegated) targetVerbState else postSubjectNegation(targetVerbState)
    progress(
      "" -> pure(subj(targetVerbState, alreadyNegated)),
      "" -> markAnswerSlot(Subj, makeNounForWh)
        .as(skipSubjTarget) // can skip directly to verb if we make subj the answer
    )
  }

  def negContraction(subjRequired: Boolean, targetVerbState: TemplateState) = {
    def target(negate: Boolean) =
      if (subjRequired) subj(targetVerbState, negate) else optionalSubj(targetVerbState, negate)
    progress(
      ""    -> pure(target(false)),
      "n't" -> modify((FrameState.frame composeLens Frame.isNegated).set(true)).as(target(true))
    )
  }

  def haveAux(subjRequired: Boolean) = {
    val target = negContraction(subjRequired, pastParticipleVerb)
    progress(
      " has" -> modFrame(Frame.tense.set(Tense.Finite.Present) andThen Frame.isPerfect.set(true))
        .as(target),
      " had" -> modFrame(Frame.tense.set(Tense.Finite.Past) andThen Frame.isPerfect.set(true)).as(target)
    )
  }

  def infNegContraction(subjRequired: Boolean) = negContraction(subjRequired, infinitiveVerb)

  def modalAux(subjRequired: Boolean) = {
    def infSubj(negate: Boolean) =
      if (subjRequired) subj(infinitiveVerb, negate) else optionalSubj(infinitiveVerb, negate)
    val infNegContraction = negContraction(subjRequired, infinitiveVerb)
    progress(
      " can't" -> modFrame(
        Frame.tense.set(Tense.Finite.Modal("can".lowerCase)) andThen Frame.isNegated.set(true)
      ).as(infSubj(true)),
      " can" -> modFrame(Frame.tense.set(Tense.Finite.Modal("can".lowerCase))).as(infSubj(false)),
      " won't" -> modFrame(
        Frame.tense.set(Tense.Finite.Modal("will".lowerCase)) andThen Frame.isNegated.set(true)
      ).as(infSubj(true)),
      " will"   -> modFrame(Frame.tense.set(Tense.Finite.Modal("will".lowerCase))).as(infSubj(false)),
      " might"  -> modFrame(Frame.tense.set(Tense.Finite.Modal("might".lowerCase))).as(infSubj(false)),
      " would"  -> modFrame(Frame.tense.set(Tense.Finite.Modal("would".lowerCase))).as(infNegContraction),
      " should" -> modFrame(Frame.tense.set(Tense.Finite.Modal("should".lowerCase))).as(infNegContraction)
    )
  }

  def doAux(subjRequired: Boolean) = {
    val target = negContraction(subjRequired, stemVerb)
    progress(
      " does" -> modFrame(Frame.tense.set(Tense.Finite.Present)).as(target),
      " did"  -> modFrame(Frame.tense.set(Tense.Finite.Past)).as(target)
    )
  }

  def beAux(subjRequired: Boolean) = {
    val target = negContraction(subjRequired, presentParticipleOrPassiveVerb)
    progress(
      " is"  -> modFrame(Frame.tense.set(Tense.Finite.Present)).as(target),
      " was" -> modFrame(Frame.tense.set(Tense.Finite.Past)).as(target)
    )
  }

  def preAux(subjRequired: Boolean) = {
    val tail = NonEmptyList.of[TemplateTransition](
      "" -> pure(beAux(subjRequired)),
      "" -> pure(modalAux(subjRequired)),
      "" -> pure(doAux(subjRequired)),
      "" -> pure(haveAux(subjRequired))
    )
    val straightToVerb: TemplateTransition =
      "" -> markAnswerSlot(Subj, makeNounForWh).as(tensedVerb)
    val transitions = if (subjRequired) tail else straightToVerb :: tail
    TemplateProgress(transitions)
  }

  def adverbial(adv: String) = for {
    _ <- modify(FrameState.whWord.set(Some(adv.lowerCase)))
    _ <- modify(FrameState.answerSlot.set(Some(Adv(adv.lowerCase))))
  } yield preAux(subjRequired = true): TemplateState

  val wh = {
    val aux = preAux(subjRequired = false)
    val auxRequiringSubject = preAux(subjRequired = true)
    TemplateProgress(
      NonEmptyList.of[TemplateTransition](
        "Who"   -> modify(FrameState.whWord.set(Some("who".lowerCase))).as(aux),
        "What"  -> modify(FrameState.whWord.set(Some("what".lowerCase))).as(aux),
        "Where" -> modify(FrameState.whWord.set(Some("where".lowerCase))).as(auxRequiringSubject)
      ).concat(
        List("Where", "When", "Why", "How", "How much", "How long").map(adv => adv -> adverbial(adv))
      )
    )
  }

  def start = wh
}
