package qfirst.clause.ext

import jjm.LowerCaseString
import jjm.ling.en.InflectedForms
import jjm.implicits._

import cats.data.NonEmptyList
import cats.data.EitherT
import cats.implicits._

import VerbTemplateStateMachine.TemplateState
import VerbTemplateStateMachine.VerbState

import monocle.Iso

class VerbStringProcessor(stateMachine: VerbTemplateStateMachine) {

  import VerbStringProcessor._

  def getStatesFromTransition(
    textSoFarReversed: List[Char],
    verbState: VerbState,
    newState: TemplateState
  ): List[ValidState] = newState match {
    case VerbTemplateStateMachine.TemplateComplete =>
      List(CompleteState(textSoFarReversed.reverse.mkString, verbState.subj, verbState.isPassive, verbState.tan))
    case VerbTemplateStateMachine.TemplateProgress(transitions) =>
      transitions.toList.flatMap {
        case (token, nextStateProcessor) =>
          nextStateProcessor.run(verbState) match {
            case None => Nil
            case Some((newVerbState, nextState)) =>
              NonEmptyList.fromList(token.toList) match {
                case None => getStatesFromTransition(textSoFarReversed, newVerbState, nextState)
                case Some(chars) =>
                  List(
                    InProgressState(textSoFarReversed, newVerbState, chars, nextState)
                  )
              }
          }
      }
  }

  def processCharacter(state: ValidState, observedChar: Char): ProcessingState = state match {
    case c @ CompleteState(str, _, _, _) =>
      EitherT.left[ValidState](List(InvalidState(c, str.size)))
    case ips @ InProgressState(
          textSoFarReversed,
          verbState,
          textRemainingInCurrentTransition,
          targetState
        ) =>
      val expectedChar = textRemainingInCurrentTransition.head
      if (expectedChar.toLower != observedChar.toLower) {
        EitherT.left[ValidState](
          List(InvalidState(ips, textSoFarReversed.size))
        )
      } else {
        val newTextReversed = expectedChar :: textSoFarReversed
        NonEmptyList.fromList(textRemainingInCurrentTransition.tail) match {
          case None =>
            EitherT.right[InvalidState](
              getStatesFromTransition(newTextReversed, verbState, targetState)
            )
          case Some(remainingChars) =>
            EitherT.right[InvalidState](
              List(
                InProgressState(newTextReversed, verbState, remainingChars, targetState)
              )
            )
        }
      }
  }

  def processString(input: String): ProcessingState = {
    // TODO WHY IS THIS NOT THE SAME AS BELOW?
    // EitherT.right[NonEmptyList, InvalidState, ValidState](getStatesFromTransition(Nil, 0)) >>= (init =>
    //   input.toList.foldM[EitherT[NonEmptyList, InvalidState, ?], ValidState](init)(processCharacter)
    // )

    // the equality I expected:
    // mz >>= (z => l.foldLeftM(z)(f)) == l.foldLeft(mz)((ma, x) => ma >>= (f(_, x)))

    input.toList.foldLeft(
      EitherT.right[InvalidState](
        getStatesFromTransition(Nil, VerbTemplateStateMachine.VerbState.initial, stateMachine.start)
      )
    ) {
      case (acc, char) =>
        acc.flatMap(processCharacter(_, char))

      // val firstNecessarilyInvalidCharOpt = acc.value.toList.collect {
      //   case Left(InvalidState(_, n)) => n
      // }.maximumOption

      // NOTE pre-exludes bad early results for efficiency. could put in another impl. but right now doesn't matter
      // val newAcc = firstNecessarilyInvalidCharOpt.fold(acc) { firstNecessarilyInvalidChar =>
      //   EitherT(
      //     NonEmptyList.fromList(
      //       acc.value.toList.filter {
      //         case Left(InvalidState(_, n)) => n == firstNecessarilyInvalidChar
      //         case _ => true
      //       }).get
      //     )
      // }
      // newAcc.flatMap(processCharacter(_, char))
    }
  }

  def processStringFully(
    input: String
  ): Either[AggregatedInvalidState, NonEmptyList[ValidState]] = {
    val resultState = processString(input)
    // we know that that at least one of the two parts is nonempty
    val (invalidStates, validStates) = resultState.value.toList.separate
    Either.fromOption(
      NonEmptyList.fromList(validStates), {
        // so now we know validStates is empty
        val maxGoodLength = invalidStates.map(_.numGoodCharacters).max
        // and this as well, since something had to have the max
        val lastGoodStates = invalidStates.collect {
          case InvalidState(lastGoodState, `maxGoodLength`) => lastGoodState
        }
        AggregatedInvalidState(NonEmptyList.fromList(lastGoodStates).get, maxGoodLength)
      }
    )
  }

  def isValid(input: String): Boolean =
    processStringFully(input).toOption.exists(_.exists(_.isComplete))

  // TODO move
  def isAlmostComplete(state: InProgressState) =
    state.targetState == VerbTemplateStateMachine.TemplateComplete

}

object VerbStringProcessor {

  type ProcessingState = EitherT[List, InvalidState, ValidState]

  sealed trait ValidState {
    def fullText: String
    def isComplete: Boolean
  }

  object ValidState {

    def eitherIso: Iso[ValidState, Either[InProgressState, CompleteState]] =
      Iso[ValidState, Either[InProgressState, CompleteState]](
        vs =>
          vs match {
            case ips: InProgressState => Left(ips)
            case cs: CompleteState    => Right(cs)
        }
      )(
        eith =>
          eith match {
            case Left(ips) => ips: ValidState
            case Right(cs) => cs: ValidState
        }
      )
  }

  case class CompleteState(
    override val fullText: String,
    subj: Option[Noun],
    isPassive: Boolean,
    tan: TAN
  ) extends ValidState {
    def isComplete = true
  }
  case class InProgressState(
    textSoFarReversed: List[Char],
    verbState: VerbState,
    textRemainingInCurrentTransition: NonEmptyList[Char],
    targetState: VerbTemplateStateMachine.TemplateState
  ) extends ValidState {
    override def fullText =
      textSoFarReversed.reverse.mkString + textRemainingInCurrentTransition.toList.mkString
    override def isComplete = false
  }

  case class InvalidState(lastGoodState: ValidState, numGoodCharacters: Int)

  case class AggregatedInvalidState(
    lastGoodStates: NonEmptyList[ValidState],
    numGoodCharacters: Int
  )

}
