package qfirst.clause.ext

import jjm.DependentMap
import jjm.LowerCaseString
import jjm.ling.en._
import jjm.ling.en.VerbForm._
import jjm.implicits._

import cats.Id
import cats.Foldable
import cats.data.NonEmptyList
import cats.data.StateT
import cats.data.State
import cats.implicits._

import monocle.macros._

import io.circe.generic.JsonCodec

// TODO: restrict argument slots to non-adv
@JsonCodec @Lenses case class ArgStructure(
  args: DependentMap[ArgumentSlot.Aux, Id],
  isPassive: Boolean
) {
  def forgetAnimacy = {
    val nounLikeAnim = NounLikeArgument.noun
      .composeLens(Noun.isAnimate)
    val prepObjAnim = Preposition.objOpt
      .composePrism(monocle.std.option.some)
      .composeOptional(nounLikeAnim)
    val miscObjAnim = NonPrepArgument.nounLikeArgument
      .composeOptional(nounLikeAnim)

    val newArgs = args.keys.foldLeft(DependentMap.empty[ArgumentSlot.Aux, Id]) {
      (m, k) => k match {
        case Adv(wh) => m.put(Adv(wh), args.get(Adv(wh)).get)
        case Subj   => m.put(Subj, Noun.isAnimate.set(false)(args.get(Subj).get))
        case Obj    => m.put(Obj, Noun.isAnimate.set(false)(args.get(Obj).get))
        case Prep1  => m.put(Prep1, prepObjAnim.set(false)(args.get(Prep1).get))
        case Prep2  => m.put(Prep2, prepObjAnim.set(false)(args.get(Prep2).get))
        case Misc   => m.put(Misc, miscObjAnim.set(false)(args.get(Misc).get))
      }
    }
    this.copy(args = newArgs)
  }
}
object ArgStructure

@JsonCodec @Lenses case class TAN(
  tense: Tense,
  isPerfect: Boolean,
  isProgressive: Boolean,
  isNegated: Boolean
) {
  def getVerbPrefixAndForm(
    isPassive: Boolean,
    subjectPresent: Boolean
  ): (List[LowerCaseString], VerbForm) = {
    val dummyFrame = Frame(
      ArgStructure(DependentMap.empty[ArgumentSlot.Aux, Id], isPassive),
      InflectedForms.generic, this)
    val initVerbStack = dummyFrame.getVerbStack
    val verbStack = if(subjectPresent) {
      dummyFrame.splitVerbStackIfNecessary(initVerbStack)
    } else initVerbStack
    val verbPrefix = verbStack.init.map(_.lowerCase)
    val verbForm = dummyFrame.getVerbConjugation(subjectPresent)
    verbPrefix -> verbForm
  }
}
object TAN

@JsonCodec @Lenses case class Frame(
  structure: ArgStructure,
  verbInflectedForms: InflectedForms,
  tan: TAN
) {

  @inline def args = structure.args
  @inline def isPassive = structure.isPassive
  @inline def tense = tan.tense
  @inline def isPerfect = tan.isPerfect
  @inline def isProgressive = tan.isProgressive
  @inline def isNegated = tan.isNegated

  private[this] def modalTokens(modal: LowerCaseString) =
    if (isNegated) {
      if (modal.toString == "will") NonEmptyList.of("won't")
      else if (modal.toString == "can") NonEmptyList.of("can't")
      else if (modal.toString == "might") NonEmptyList.of("might", "not")
      else NonEmptyList.of(s"${modal}n't")
    } else {
      NonEmptyList.of(modal.toString)
    }

  private[this] def getForms(s: LowerCaseString) = {
    if (verbInflectedForms.allForms.contains(s)) Some(verbInflectedForms)
    else if (InflectedForms.beSingularForms.allForms.contains(s)) Some(InflectedForms.beSingularForms)
    else if (InflectedForms.doForms.allForms.contains(s)) Some(InflectedForms.doForms)
    else if (InflectedForms.haveForms.allForms.contains(s)) Some(InflectedForms.haveForms)
    else None
  }

  private[this] def push(s: String) =
    State.modify[NonEmptyList[String]](s :: _)
  private[this] def pushAll(ss: NonEmptyList[String]) =
    State.modify[NonEmptyList[String]](x => ss ++ x.toList)
  private[this] def modTop(f: String => String) =
    State.modify[NonEmptyList[String]](l => NonEmptyList(f(l.head), l.tail))
  private[this] def modForm(form: VerbForm) =
    modTop(w => getForms(w.lowerCase).fold(w)(_(form)))

  // should always agree with what's produced on the verb stack.
  // ideally they would share code somehow but this is easiest for now and probably works.
  def getVerbConjugation(subjectPresent: Boolean): VerbForm = {
    if (isPassive) PastParticiple
    else if (isProgressive) PresentParticiple
    else if (isPerfect) PastParticiple
    else
      tense match {
        case Tense.Finite.Modal(_)              => Stem
        case _ if (isNegated || subjectPresent) => Stem
        case Tense.Finite.Past                  => Past
        case Tense.Finite.Present               => PresentSingular3rd
        case Tense.NonFinite.Bare               => Stem
        case Tense.NonFinite.To                 => Stem
        case Tense.NonFinite.Gerund             => PresentParticiple
      }
  }

  def getVerbStack = {
    def pass = State.pure[NonEmptyList[String], Unit](())

    val stackState = for {
      // start with verb stem
      _               <- (if (isPassive) modForm(PastParticiple) >> push("be") else pass)
      _               <- (if (isProgressive) modForm(PresentParticiple) >> push("be") else pass)
      _               <- (if (isPerfect) modForm(PastParticiple) >> push("have") else pass)
      postAspectStack <- State.get[NonEmptyList[String]]
      _ <- tense match {
        case Tense.Finite.Modal(m) => pushAll(modalTokens(m))
        case Tense.Finite.Past =>
          if (isNegated) {
            if (postAspectStack.size == 1) push("didn't")
            else (modForm(Past) >> modTop(_ + "n't"))
          } else modForm(Past)
        case Tense.Finite.Present =>
          if (isNegated) {
            if (postAspectStack.size == 1) push("doesn't")
            else (modForm(PresentSingular3rd) >> modTop(_ + "n't"))
          } else modForm(PresentSingular3rd)
        case nf: Tense.NonFinite =>
          val verbMod = nf match {
            case Tense.NonFinite.Bare => pass
            case Tense.NonFinite.To => push("to")
            case Tense.NonFinite.Gerund => modForm(PresentParticiple)
          }
          verbMod >> (if (isNegated) push("not") else pass)
      }
    } yield ()

    stackState.runS(NonEmptyList.of(verbInflectedForms.stem)).value
  }

  def splitVerbStackIfNecessary(verbStack: NonEmptyList[String]) = {
    if (verbStack.size > 1) {
      verbStack
    } else
      tense match {
        case Tense.Finite.Past     => (modForm(Stem) >> push("did")).runS(verbStack).value
        case Tense.Finite.Present  => (modForm(Stem) >> push("does")).runS(verbStack).value
        case Tense.Finite.Modal(_) => verbStack // should never happen, since a modal adds another token
        case _ => verbStack // Non-finite case, where splitting cannot occur
      }
  }

  private[this] def append[A](a: A): StateT[List, List[Either[String, A]], Unit] =
    StateT.modify[List, List[Either[String, A]]](Right(a) :: _)
  private[this] def appendString[A](word: String): StateT[List, List[Either[String, A]], Unit] =
    StateT.modify[List, List[Either[String, A]]](Left(word) :: _)
  private[this] def appendEither[A](e: Either[String, A]): StateT[List, List[Either[String, A]], Unit] =
    StateT.modify[List, List[Either[String, A]]](e :: _)
  private[this] def appendAllStrings[F[_]: Foldable, A](fs: F[String]): StateT[List, List[Either[String, A]], Unit] =
    fs.foldM[StateT[List, List[Either[String, A]], *], Unit](()) { case (_, s) => appendString(s) }
  private[this] def appendAll[F[_]: Foldable, A](fs: F[Either[String, A]]): StateT[List, List[Either[String, A]], Unit] =
    fs.foldM[StateT[List, List[Either[String, A]], *], Unit](()) { case (_, s) => appendEither(s) }
  private[this] def choose[A, B](as: List[A]): StateT[List, List[Either[String, B]], A] =
    StateT.liftF[List, List[Either[String, B]], A](as)
  private[this] def pass[A]: StateT[List, List[Either[String, A]], Unit] =
    StateT.pure[List, List[Either[String, A]], Unit](())
  // private[this] def abort[A]: StateT[List, List[Either[String, A]], Unit] =
  //   choose[Unit, A](List[Unit]())

  type ArgMap[A] = Map[ArgumentSlot, A]

  private[this] def renderNecessaryNoun[A](slot: ArgumentSlot.Aux[Noun], argValues: ArgMap[A]) = args.get(slot) match {
    case None       => choose[String, A](List("someone", "something")) >>= appendString[A]
    case Some(noun) => argValues.get(slot).fold(appendAllStrings[List, A](noun.placeholder))(append[A])
  }

  private[this] def renderWhNoun[A](slot: ArgumentSlot.Aux[Noun]) = args.get(slot) match {
    case None       => choose[String, A](List("Who", "What")) >>= appendString[A]
    case Some(noun) => choose[String, A](noun.wh.toList) >>= appendString[A]
  }

  private[this] def renderWhOrAbort[Arg <: Argument, A](slot: ArgumentSlot.Aux[Arg]) =
    choose[String, A]((args.get(slot) >>= (_.wh)).toList) >>= appendString[A]

  private[this] def renderArgIfPresent[Arg <: Argument, A](slot: ArgumentSlot.Aux[Arg], argValues: ArgMap[A]) =
    args.get(slot).fold(pass[A])(argSlotValue =>
      argValues.get(slot).fold(appendAllStrings[List, A](argSlotValue.unGap ++ argSlotValue.placeholder))(argMapValue =>
        appendAll(argSlotValue.unGap.map(Left[String, A](_)) ++ List(Right[String, A](argMapValue)))
      )
    )

  private[this] def renderGap[Arg <: Argument, A](slot: ArgumentSlot.Aux[Arg]) =
    appendAllStrings[List, A](args.get(slot).toList >>= (_.gap))

  private[this] def renderAuxThroughVerb[A](includeSubject: Boolean, argValues: ArgMap[A]) = {
    val verbStack = getVerbStack
    if (includeSubject) {
      val splitVerbStack = splitVerbStackIfNecessary(verbStack)
      val (aux, verb) = (splitVerbStack.head, splitVerbStack.tail)
      appendString[A](aux) >> renderNecessaryNoun(Subj, argValues) >> appendAllStrings[List, A](verb)
    } else appendAllStrings[NonEmptyList, A](verbStack)
  }

  def questionsForSlot(slot: ArgumentSlot) = questionsForSlotWithArgs(slot, Map[ArgumentSlot, String]())

  def clauses(addParens: Boolean = false) = clausesWithArgs(Map(), addParens)

  def clausesWithArgs(argValues: ArgMap[String], addParens: Boolean = false) = {
    val qStateT = {
      renderNecessaryNoun(Subj, argValues) >>
        renderAuxThroughVerb(includeSubject = false, argValues) >>
        renderArgIfPresent(Obj  , argValues) >>
        (
          if(addParens) {
            append("(") >> renderArgIfPresent(Prep1, argValues) >> append(")") >>
            append("(") >> renderArgIfPresent(Prep2, argValues) >> append(")") >>
              append("(") >> renderArgIfPresent(Misc , argValues) >> append(")")
          } else {
            renderArgIfPresent(Prep1, argValues) >>
              renderArgIfPresent(Prep2, argValues) >>
              renderArgIfPresent(Misc , argValues)
          }
        )
    }
    qStateT.runS(List.empty[Either[String, String]]).map(_.map(_.merge)).map(_.reverse.mkString(" "))
  }

  def genClausesWithArgs[A](
    argValues: Map[ArgumentSlot, A]
  ): List[List[Either[String, A]]] = {
    val qStateT = {
      renderNecessaryNoun(Subj, argValues) >>
        renderAuxThroughVerb(includeSubject = false, argValues) >>
        renderArgIfPresent(Obj  , argValues) >>
        renderArgIfPresent(Prep1, argValues) >>
        renderArgIfPresent(Prep2, argValues) >>
        renderArgIfPresent(Misc , argValues)
    }
    qStateT.runS(List.empty[Either[String, A]]).map(_.reverse)
  }

  def clausesWithArgMarkers = {
    genClausesWithArgs(args.keys.map(a => a.asInstanceOf[ArgumentSlot.Aux[Argument]]).map(a => a -> a).toMap)
  }

  def questionsForSlotWithArgs(slot: ArgumentSlot, argValues: ArgMap[String]): List[String] = {
    questionsForSlotWithArgs(Some(slot), argValues)
  }

  def questionsForSlotWithArgs(slotOpt: Option[ArgumentSlot], argValues: ArgMap[String]): List[String] = {
    val qStateT = slotOpt match {
      case None =>
        renderAuxThroughVerb(includeSubject = true, argValues) >>
        renderArgIfPresent(Obj  , argValues) >>
        renderArgIfPresent(Prep1, argValues) >>
        renderArgIfPresent(Prep2, argValues) >>
        renderArgIfPresent(Misc , argValues)
      case Some(slot) => slot match {
        case Subj =>
          renderWhNoun[String](Subj) >>
          renderAuxThroughVerb(includeSubject = false, argValues) >>
          renderArgIfPresent(Obj  , argValues) >>
          renderArgIfPresent(Prep1, argValues) >>
          renderArgIfPresent(Prep2, argValues) >>
          renderArgIfPresent(Misc , argValues)
        case Obj =>
          renderWhNoun[String](Obj) >>
          renderAuxThroughVerb(includeSubject = true, argValues) >>
          renderGap[Noun, String](Obj) >>
          renderArgIfPresent(Prep1, argValues) >>
          renderArgIfPresent(Prep2, argValues) >>
          renderArgIfPresent(Misc , argValues)
        case Prep1 =>
          renderWhOrAbort[Preposition, String](Prep1) >>
          renderAuxThroughVerb(includeSubject = true, argValues) >>
          renderArgIfPresent(Obj, argValues) >>
          renderGap[Preposition, String](Prep1) >>
          renderArgIfPresent(Prep2, argValues) >>
          renderArgIfPresent(Misc , argValues)
        case Prep2 =>
          renderWhOrAbort[Preposition, String](Prep2) >>
          renderAuxThroughVerb(includeSubject = true, argValues) >>
          renderArgIfPresent(Obj  , argValues) >>
          renderArgIfPresent(Prep1, argValues) >>
          renderGap[Preposition, String](Prep2) >>
          renderArgIfPresent(Misc , argValues)
        case Misc =>
          renderWhOrAbort[NonPrepArgument, String](Misc) >>
          renderAuxThroughVerb(includeSubject = true, argValues) >>
          renderArgIfPresent(Obj  , argValues) >>
          renderArgIfPresent(Prep1, argValues) >>
          renderArgIfPresent(Prep2, argValues) >>
          renderGap[NonPrepArgument, String](Misc)
        case Adv(wh) =>
          append(wh.toString.capitalize) >>
          renderAuxThroughVerb(includeSubject = true, argValues) >>
          renderArgIfPresent(Obj  , argValues) >>
          renderArgIfPresent(Prep1, argValues) >>
          renderArgIfPresent(Prep2, argValues) >>
          renderArgIfPresent(Misc , argValues)
      }
    }
    qStateT.runS(List.empty[Either[String, String]]).map(_.map(_.merge)).map(_.reverse.mkString(" ") + "?")
  }
}

object Frame {

  val args = Frame.structure composeLens ArgStructure.args
  val isPassive = Frame.structure composeLens ArgStructure.isPassive
  val tense = Frame.tan composeLens TAN.tense
  val isNegated = Frame.tan composeLens TAN.isNegated
  val isProgressive = Frame.tan composeLens TAN.isProgressive
  val isPerfect = Frame.tan composeLens TAN.isPerfect

  def empty(verbForms: InflectedForms) =
    Frame(
      ArgStructure(
        DependentMap.empty[ArgumentSlot.Aux, Id],
        false),
      verbForms,
      TAN(
        Tense.Finite.Past,
        false,
        false,
        false)
    )
}
