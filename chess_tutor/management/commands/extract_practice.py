"""Bootstrap practice from an existing verified Chess.com archive, offline."""
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from chess_tutor import practice
from chess_tutor.chesscom_import import iter_training_examples
from chess_tutor.models import PlayerProfile
from chess_tutor.opponent import EngineUnavailable
from chess_tutor.personal_profile import canonical_username, named_profile_id
from chess_tutor.practice import create_lesson
from scripts.train_history import cached_games


class Command(BaseCommand):
    help = 'Extract up to three mistakes per recent cached game into the named player’s practice queue.'

    def add_arguments(self, parser):
        parser.add_argument('--username', required=True)
        parser.add_argument('--games', type=int, default=5)
        parser.add_argument('--cache-dir', type=Path, default=settings.BASE_DIR / '.runtime/players')

    def handle(self, *args, **options):
        room = None
        try:
            username = canonical_username(options['username'])
            if not 1 <= options['games'] <= 20:
                raise ValueError('Choose between 1 and 20 recent games.')
            profile = PlayerProfile.objects.filter(pk=named_profile_id(username)).first()
            if profile is None:
                raise ValueError('Activate this player before extracting lessons.')
            imported = cached_games(username, options['cache_dir'])
            room = practice.engine_room()
            total = 0
            for record in reversed(imported.games[-options['games']:]):
                count = 0
                for example in iter_training_examples(record):
                    board = example['board']
                    move = board.parse_uci(example['move_uci'])
                    source = f"Chess.com · {record['url'].rstrip('/').split('/')[-1]} · move {board.fullmove_number}"
                    count += create_lesson(profile, board, move, source, room=room) is not None
                    if count >= 3:
                        break
                total += count
                self.stdout.write(f"Checked {record['url']}: {count} new lessons")
            self.stdout.write(self.style.SUCCESS(f'{total} lessons added. Open /practice/ to begin.'))
        except (ValueError, OSError, KeyError, EngineUnavailable) as error:
            raise CommandError(str(error)) from error
        finally:
            if room is not None:
                room.close()
