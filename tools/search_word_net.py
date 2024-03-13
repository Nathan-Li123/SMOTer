import os
from nltk.corpus import wordnet

total_relations = ['accept.v.01', 'accept.v.02', 'accompany.v.02', 'adjust.v.01', 'admire.v.01', 'agree.v.01', 'agree.v.02', 'aim.v.01', 'amuse.v.02', 'anger.v.01', 'answer.v.03', 'applaud.v.01', 'appreciate.v.01', 
                   'approach.v.01', 'render.v.04', 'arise.v.03', 'attack.v.01', 'bandage.v.02', 'baptism.n.01', 'barber.v.01', 'bark.v.01', 'barricade.v.01', 'beat.v.02', 'beat.v.03', 'beckon.v.01', 'beckon.v.03', 
                   'bite.v.01', 'bite.v.02', 'blow.v.06', 'box.v.02', 'braid.v.03', 'breathe.v.01', 'brush.v.01', 'brush.v.03', 'bubble.v.01', 'bubble.v.04', 'buy.v.01', 'call.v.22', 'caress.v.01', 
                   'catch.v.01', 'catch.v.04', 'change_state.v.01', 'chase.v.01', 'check.v.01', 'check.v.02', 'check.v.06', 'cheer.v.02', 'cheer.v.05', 'circle.v.01', 'circle.v.02', 'clamp.v.01', 
                   'clap.v.01', 'clap.v.02', 'clap.v.04', 'clap.v.06', 'clean.v.01', 'clip.v.03', 'close.v.10', 'collaborate.v.01', 'collide.v.02', 'collision', 'comb.v.01', 'command.n.02', 
                   'compel.v.01', 'compete.v.01', 'complain.v.01', 'comply.v.0', 'comply.v.01', 'compress.v.02', 'congratulate.v.02', 'converse.n.01', 'converse.v.01', 'cooperation', 'cover.v.01', 
                   'cover.v.19', 'criticize.v.01', 'cry.v.01', 'cry.v.02', 'curl.v.01', 'cut.v.07', 'dab.v.02', 'dance.v.01', 'dance.v.03', 'daub.v.03', 'decant.v.01', 'defense', 'defy.v.02', 'detach.v.01', 'die.v.10', 
                   'direct.v.09', 'disinfect.v.01', 'dodge.v.01', 'drag.v.01', 'draw.v.05', 'dress.v.01', 'dress.v.02', 'dress.v.16', 'drink.v.01', 'drip.v.01', 'drop.v.01', 'drum.v.01', 'dry.v.01', 'eat.v.01', 
                   'elude.v.01', 'embrace.v.02', 'escape.v.01', 'examine.v.02', 'exchange.v.01', 'explain.v.01', 'facial.n.02', 'feed.v.01', 'feed.v.02', 'feed.v.06', 'feel.v.03', 'fight.v.02', 'finger.v.01', 
                   'flee.v.01', 'follow.v.01', 'free.v.01', 'frighten.v.01', 'frisk.v.02', 'give.v.01', 'give.v.03', 'give.v.07', 'give.v.08', 'give.v.14', 'give.v.28', 'glance.v.01', 'glue.v.01', 'grasp.v.01', 
                   'grip.v.01', 'guffaw.v.01', 'hammer.v.01', 'handshake.n.01', 'hash_out.v.0', 'hash_out.v.01', 'headshake', 'help.v.01', 'hit.v.02', 'hit.v.03', 'hold.v.0', 
                   'hold.v.01', 'hold.v.02', 'hold.v.10', 'hold.v.13', 'hold.v.14', 'hook.v.11', 'imitate.v.01', 'imitate.v.02', 'indicate.v.02', 'insert.v.01', 'insert.v.02', 'interview.v.01', 'invite.v.04', 'invite.v.05', 
                   'judge.v.01', 'jump.v.01', 'keep.v.01', 'kid.v.01', 'kiss.v.02', 'kneel.v.01', 'knock.v.01', 'laugh.v.01', 'lead.v.01', 'lean.v.01', 'lean_on.v.01', 'leave.v.01', 'let_go_of.v.01', 'lick.v.02', 'lie.v.01', 
                   'lie.v.02', 'lift.v.0', 'lift.v.02', 'lift.v.03', 'light.v.01', 'listen.v.01', 'listen.v.02', 'look.v.01', 'look.v.1', 'lower.v.01', 'manicure.v.01', 
                   'manipulate.v.05', 'massage.v.0', 'massage.v.01', 'massage.v.02', 'measure.v.01', 'meet.v.10', 'model.v.03', 'moisten.v.01', 'moisture.n.01', 'money.n.01', 'move.v.02', 'move.v.03', 'nod.v.01', 
                   'nod.v.02', 'observe.v.04', 'obstruct.v.01', 'open.v.01', 'order.v.0', 'order.v.01', 'orient.v.02', 'parry.v.01', 'parry.v.02', 'pass.v.01', 'pass.v.05', 'pass.v.20', 'paste.v.03', 'pay.v.01', 
                   'peck.v.01', 'photograph.v.01', 'pinch.v.01', 'play.v.01', 'play.v.03', 'point.v.10', 'polish.v.02', 'pour.v.01', 'preen.v.02', 'press.v.01', 'press.v.02', 'propose.v.05', 'provide.v.02', 'pull.v.01', 
                   'pull.v.04', 'push.v.01', 'push.v.05', 'quarrel.v.01', 'question.v.03', 'quibble.v.02', 'raid.v.01', 'raise.v.02', 'rally.n.05', 'rally.v.01', 'rally.v.05', 'read.v.03', 'read.v.05', 'receive.v.01', 
                   'record.v.01', 'refuse.v.01', 'reject.v.01', 'remove.v.01', 'render.v.07', 'request.v.01', 'resist.v.01', 'resist.v.02', 'respond.v.03', 'return.v.01', 'return.v.06', 
                   'rinse.v.01', 'rise.v.01', 'rob.v.01', 'rock.v.01', 'rub.v.01', 'rub.v.02', 'run.v.01', 'salute.v.02', 'screen.v.02', 'seize.v.01', 'sell.v.01', 'serve.v.01', 'serve.v.05', 'serve.v.15', 
                   'shake.v.01', 'shake.v.08', 'sharpen.v.07', 'shave.v.01', 'shear.v.01', 'show.v.01', 'show.v.04', 'sing.v.01', 'sing.v.02', 'sit.v.01', 'sit.v.07', 
                   'smell.v.01', 'smile.v.01', 'smile.v.02', 'snog.v.01', 'spray.v.01', 'spray.v.02', 'sprinkle.v.02', 'squeeze.v.02', 'stand.v.01', 'stare.v.02', 
                   'stay.v.01', 'step.v.02', 'stick.v.08', 'stop.v.03', 'straddle.v.01', 'stretch.v.02', 'stride.v.01', 'strip.v.13', 'stroke.v.01', 'struggle.v.02', 'study.v.01', 'study.v.02', 'support.v.01', 
                   'sweep.v.02', 'swing.v.03', 'take.v.01', 'take.v.04', 'take_after.v.02', 'take_off.v.02', 'talk.v.01', 'talk.v.02', 'tap.v.07', 'taste.v.01', 'teach.v.01', 'tease.v.03', 'through', 'throw.v.01', 
                   'tidy.v.01', 'tie.v.05', 'tongue.v.02', 'touch.v.01', 'touch.v.02', 'towards', 'tread.v.02', 'turn.v.04', 'turn.v.09', 'tutor.v.01', 'undress.v.01', 'unfold.v.04', 'unrobe', 'walk.v.01', 'walk.v.08', 
                   'wash.v.01', 'wash.v.02', 'wash.v.03', 'wear.v.01', 'wear.v.02', 'wear.v.05', 'wear.v.09', 'wheedle.v.01', 'wipe.v.01', 'wrap.v.01', 'write.v.02']


if __name__ == '__main__':
    f = open('./tools/interactions.txt', 'w')
    total_relations.sort()
    for relation in total_relations:
        if relation in ['collision', 'cooperation', 'defense', 'headshake', 'towards', 'recieve', 'through', 'unrobe']:
            f.write('\\noindent\n\\textbf{%s: }\n\n' % relation)
            continue
        # text = relation + ': ' + wordnet.synset(relation).definition()
        f.write('\\noindent\n\\textbf{%s: }\emph{%s}\n\n' % (relation, wordnet.synset(relation).definition()))
    f.close()
